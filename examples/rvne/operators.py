"""Execute reusable integer SNN operators with NineToothed and RVNE QEMU."""

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import ninetoothed.language as ntl
from ninetoothed import Tensor
from ninetoothed.compiler import CompileRequest, compile_kernel
from ninetoothed.compiler.runtime import materialize


def _identity(*tensors):
    return tensors


def _conv2d_application(patches, weight_low, weight_high, out):
    current = ntl.zeros(out.shape, dtype=ntl.int32)

    for block in range(patches.shape[0]):
        spikes = patches[block][None, :]
        low = weight_low[block][:, None]
        high = weight_high[block][:, None]
        current = ntl.spike_accumulate(current, spikes, low, high)

    out = current  # noqa: F841


def _max_pool2d_application(samples, out):
    top = ntl.maximum(samples[0], samples[1])
    bottom = ntl.maximum(samples[2], samples[3])
    out = ntl.maximum(top, bottom)  # noqa: F841


def _lif_application(current, voltage, syn_current, spike, next_voltage, next_current):
    integrated = syn_current + current
    potential = voltage - (voltage >> 10)
    potential = potential + (integrated >> 3)
    decayed = integrated - (integrated >> 4)
    fired = potential > 1
    spike = ntl.where(fired, 1, 0)  # noqa: F841
    next_voltage = ntl.where(fired, 0, potential)  # noqa: F841
    next_current = ntl.where(fired, 0, decayed)  # noqa: F841


@dataclass(frozen=True)
class PreparedWeights:
    """Keep packed OIHW weights independent of batch and spatial dimensions."""

    shape: tuple[int, int, int, int]
    low: np.ndarray
    high: np.ndarray

    @property
    def weight_bytes(self):
        """Return the shared packed weight storage in bytes."""
        return self.low.nbytes + self.high.nbytes

    @property
    def packed_nbytes(self):
        """Return the shared packed weight storage in bytes."""
        return self.weight_bytes

    @property
    def nbytes(self):
        """Return the shared packed weight storage in bytes."""
        return self.weight_bytes


class RvneOperators:
    """Compile and cache QEMU kernels; use the host only for data preparation."""

    def __init__(self, output_dir, toolchain_root=None):
        self.output_dir = Path(output_dir)
        self.toolchain_root = None if toolchain_root is None else str(toolchain_root)
        self.launch_counts = {"conv2d": 0, "max_pool2d": 0, "lif": 0}
        self.artifact_paths = []
        self.artifacts = []
        self.staging_records = []
        self.last_staging = {}
        self.max_staging_bytes = 0
        self._handles = {}

    @property
    def launch_count(self):
        """Return the number of successfully completed QEMU invocations."""
        return sum(self.launch_counts.values())

    @property
    def staging_bytes(self):
        """Return the latest conservative host staging estimate, excluding RSS."""
        return self.last_staging.get("total_bytes", 0)

    def prepare_weights(self, weights):
        """Pack signed INT4 OIHW integers once for reuse across spatial shapes."""
        weights = _integer_array(weights, "weights", ndim=4)

        if np.any(weights < -8) or np.any(weights > 7):
            raise ValueError("RVNE convolution weights must be integers in [-8, 7].")

        channels = weights.shape[0]
        flattened = weights.reshape(channels, -1)
        blocks = (flattened.shape[1] + 31) // 32
        low = np.zeros((blocks, channels), dtype=np.uint64)
        high = np.zeros_like(low)

        for lane in range(32):
            values = flattened[:, lane::32].T.astype(np.uint64)
            np.bitwise_and(values, np.uint64(15), out=values)
            np.left_shift(values, np.uint64((lane % 16) * 4), out=values)
            destination = low if lane < 16 else high
            np.bitwise_or(
                destination[: values.shape[0]],
                values,
                out=destination[: values.shape[0]],
            )

        low.flags.writeable = False
        high.flags.writeable = False

        return PreparedWeights(tuple(weights.shape), low, high)

    def conv2d(self, inputs, weights, stride=1, padding=0):
        """Convolve binary NCHW input and signed INT4 OIHW weights through RVNE."""
        inputs = _integer_array(inputs, "inputs", ndim=4)

        if np.any((inputs != 0) & (inputs != 1)):
            raise ValueError("RVNE convolution input must contain only zero and one.")

        prepared = (
            weights
            if isinstance(weights, PreparedWeights)
            else self.prepare_weights(weights)
        )
        stride = _pair(stride, "stride", minimum=1)
        padding = _pair(padding, "padding", minimum=0)
        batch, channels, height, width = inputs.shape
        output_channels, expected_channels, kernel_height, kernel_width = prepared.shape

        if channels != expected_channels:
            raise ValueError("RVNE convolution input and weight channels must match.")

        output_height = (height + 2 * padding[0] - kernel_height) // stride[0] + 1
        output_width = (width + 2 * padding[1] - kernel_width) // stride[1] + 1

        if output_height <= 0 or output_width <= 0:
            raise ValueError("RVNE convolution requires a nonempty output shape.")

        patches, packing_temporary = _pack_patches(
            inputs,
            (kernel_height, kernel_width),
            stride,
            padding,
            (output_height, output_width),
        )
        output = np.empty(
            (output_channels, batch * output_height * output_width), dtype=np.int32
        )
        runtime = self._run(
            "conv2d",
            _conv2d_application,
            (patches, prepared.low, prepared.high, output),
        )
        result = np.ascontiguousarray(
            output.reshape(
                output_channels, batch, output_height, output_width
            ).transpose(1, 0, 2, 3)
        )
        layout_copy_bytes = 0 if np.shares_memory(result, output) else result.nbytes
        self._record_staging(
            "conv2d",
            runtime,
            patches_bytes=patches.nbytes,
            weights_bytes=prepared.weight_bytes,
            output_bytes=output.nbytes,
            packing_temporary_bytes=packing_temporary,
            layout_copy_bytes=layout_copy_bytes,
        )

        return result

    def max_pool2d(self, inputs):
        """Run a 2x2, stride-two max pool; gather its four inputs on the host."""
        inputs = np.asarray(inputs)

        if inputs.ndim != 4 or any(dim <= 0 for dim in inputs.shape):
            raise ValueError("RVNE max_pool2d requires nonempty NCHW input.")

        if inputs.dtype.name not in {
            "bool",
            "int32",
            "uint32",
            "int64",
            "uint64",
            "float32",
        }:
            raise TypeError("RVNE max_pool2d received an unsupported input dtype.")

        batch, channels, height, width = inputs.shape
        output_height, output_width = height // 2, width // 2

        if output_height == 0 or output_width == 0:
            raise ValueError("RVNE max_pool2d requires both spatial dimensions >= 2.")

        samples = np.stack(
            tuple(
                inputs[
                    :,
                    :,
                    row : row + 2 * output_height : 2,
                    col : col + 2 * output_width : 2,
                ]
                for row, col in ((0, 0), (0, 1), (1, 0), (1, 1))
            )
        ).reshape(4, -1)
        output = np.empty(samples.shape[1], dtype=inputs.dtype)
        runtime = self._run("max_pool2d", _max_pool2d_application, (samples, output))
        self._record_staging(
            "max_pool2d",
            runtime,
            samples_bytes=samples.nbytes,
            output_bytes=output.nbytes,
        )

        return output.reshape(batch, channels, output_height, output_width)

    def lif(self, current, voltage, syn_current):
        """Advance int32 LIF state with shifts 10/3/4 and strict threshold one."""
        inputs = tuple(np.asarray(value) for value in (current, voltage, syn_current))

        if any(value.dtype != np.dtype("int32") for value in inputs):
            raise TypeError("RVNE LIF current and state must have int32 dtype.")

        if any(value.shape != inputs[0].shape for value in inputs):
            raise ValueError("RVNE LIF current and state shapes must match.")

        if inputs[0].ndim == 0 or inputs[0].size == 0:
            raise ValueError("RVNE LIF requires nonempty tensor state.")

        outputs = tuple(np.empty(inputs[0].shape, dtype=np.int32) for _ in range(3))
        runtime = self._run("lif", _lif_application, (*inputs, *outputs))
        self._record_staging(
            "lif",
            runtime,
            input_bytes=sum(value.nbytes for value in inputs),
            output_bytes=sum(value.nbytes for value in outputs),
        )

        return outputs

    def _run(self, name, application, arrays):
        normalized = []
        copy_bytes = 0

        for value in arrays:
            array = np.ascontiguousarray(value)

            if any(np.shares_memory(array, previous) for previous in normalized):
                array = array.copy()

            if not np.shares_memory(array, value):
                copy_bytes += array.nbytes

            normalized.append(array)

        signature = (
            name,
            tuple((value.shape, value.dtype.str) for value in normalized),
        )
        compile_seconds = 0.0

        if signature not in self._handles:
            started = time.perf_counter()
            options = (
                {}
                if self.toolchain_root is None
                else {"toolchain_root": self.toolchain_root}
            )
            compilation = compile_kernel(
                CompileRequest(
                    arrangement=_identity,
                    application=application,
                    tensors=tuple(
                        Tensor(shape=value.shape, dtype=value.dtype.name)
                        for value in normalized
                    ),
                    backend="rvne",
                    caller="numpy",
                    kernel_name=f"rvne_{name}_{len(self._handles)}",
                    backend_options=options,
                )
            )
            handle = materialize(compilation, output_dir=self.output_dir, mode="aot")
            built = handle._built_artifact

            with Path(built.binary_path).open("rb") as executable:
                header = executable.read(20)

            if (
                header[:6] != b"\x7fELF\x02\x01"
                or int.from_bytes(header[18:20], "little") != 243
            ):
                raise RuntimeError("RVNE model operator did not produce an RV64 ELF.")

            self._handles[signature] = handle
            self.artifact_paths.append(built.binary_path)
            self.artifacts.append(
                {
                    "operator": name,
                    "source": built.source_path,
                    "binary": built.binary_path,
                    "manifest": built.manifest_path,
                    "cache_key": built.cache_key,
                    "backend": compilation.artifact.backend.value,
                    "compute_arch": compilation.target.compute_arch,
                    "elf_machine": 243,
                    "elf_class": 64,
                    "source_route": compilation.artifact.metadata.get("source_route"),
                    "uses_spike_accumulator": "__builtin_riscv_calc_acc_32"
                    in compilation.artifact.primary_source,
                }
            )
            compile_seconds = time.perf_counter() - started

        started = time.perf_counter()
        self._handles[signature](*normalized)
        self.launch_counts[name] += 1

        return {
            "normalization_copy_bytes": copy_bytes,
            "compile_seconds": compile_seconds,
            "execution_seconds": time.perf_counter() - started,
        }

    def _record_staging(self, name, runtime, **sizes):
        record = {"operator": name, **runtime, **sizes}
        record["total_bytes"] = (
            sum(sizes.values()) + runtime["normalization_copy_bytes"]
        )
        self.last_staging = record
        self.staging_records.append(record)
        self.max_staging_bytes = max(self.max_staging_bytes, record["total_bytes"])


def _integer_array(values, name, *, ndim):
    values = np.asarray(values)

    if values.ndim != ndim or any(dim <= 0 for dim in values.shape):
        raise ValueError(f"RVNE {name} requires a nonempty rank-{ndim} array.")

    if values.dtype.kind not in "biu":
        raise TypeError(f"RVNE {name} requires integer or boolean values.")

    return values


def _pair(value, name, *, minimum):
    pair = (value, value) if isinstance(value, (int, np.integer)) else tuple(value)

    if len(pair) != 2 or any(
        not isinstance(item, (int, np.integer))
        or isinstance(item, (bool, np.bool_))
        or item < minimum
        for item in pair
    ):
        raise ValueError(f"RVNE {name} requires two integers >= {minimum}.")

    return tuple(int(item) for item in pair)


def _pack_patches(inputs, kernel, stride, padding, output_shape):
    batch, channels, height, width = inputs.shape
    output_height, output_width = output_shape
    positions = batch * output_height * output_width
    blocks = (channels * kernel[0] * kernel[1] + 31) // 32
    patches = np.zeros((blocks, positions), dtype=np.uint32)
    padded = np.zeros(
        (batch, channels, height + 2 * padding[0], width + 2 * padding[1]),
        dtype=np.uint8,
    )
    padded[:, :, padding[0] : padding[0] + height, padding[1] : padding[1] + width] = (
        inputs
    )
    synapse = 0

    for channel in range(channels):
        for row in range(kernel[0]):
            for col in range(kernel[1]):
                values = np.array(
                    padded[
                        :,
                        channel,
                        row : row + output_height * stride[0] : stride[0],
                        col : col + output_width * stride[1] : stride[1],
                    ],
                    dtype=np.uint32,
                    order="C",
                ).reshape(-1)
                np.left_shift(values, np.uint32(synapse % 32), out=values)
                np.bitwise_or(
                    patches[synapse // 32], values, out=patches[synapse // 32]
                )
                synapse += 1

    return patches, padded.nbytes + positions * np.dtype("uint32").itemsize


__all__ = ["PreparedWeights", "RvneOperators"]
