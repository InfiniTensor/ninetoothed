RVNE Backend
============

The ``rvne`` backend emits C++ for the RISC-V Zne extension and compiles it
with the RVNE SDK. AOT handles execute through the SDK's user-mode QEMU using
NumPy host arrays. Source generation does not need the SDK.

The current backend provides serial RISC-V arithmetic, control flow and
reductions, plus an explicit hardware spike accumulator. Integer LIF updates
can be expressed with ordinary arithmetic, shifts and ``ntl.where`` and use
explicit voltage/current input and output tensors. LIF currently uses scalar
RISC-V arithmetic, rather than the dedicated hardware LIF instruction.

Environment
-----------

Use Linux x86_64, including WSL, with the supplied RVNE SDK. Its directory must
contain ``llvm/bin/clang++`` (or ``clang``), ``qemu/bin/qemu-riscv64`` and
``xuantie-gnu-toolchain/sysroot``. This implementation was validated with the
SDK reporting Clang 15.0.6 and QEMU 6.0.94, using
``--target=riscv64-unknown-linux-gnu -march=rv64imafcvzne``.

.. code-block:: bash

   export NINETOOTHED_RVNE_TOOLCHAIN=/path/to/toolchain

Alternatively, pass ``toolchain_root="/path/to/toolchain"`` to the compiler
entrypoint. The SDK is an external dependency and is not distributed with
NineToothed. Existing Triton package dependencies still apply on the compiler
host; the generated RISC-V executable does not require Triton or PyTorch.

AOT execution
-------------

Save the application in a Python file so the compiler can inspect its source:

.. code-block:: python

   import numpy as np
   from ninetoothed import Tensor
   from ninetoothed.compiler import aot
   from ninetoothed.compiler.runtime import load_built_artifact

   def affine(
       x: Tensor(shape=(4,), dtype="int32"),
       out: Tensor(shape=(4,), dtype="int32"),
   ):
       out = x * 3 + 7

   kernel = aot(affine, backend="rvne", output_dir="build/rvne")
   x = np.array([-2, 0, 3, 10], dtype=np.int32)
   out = np.empty_like(x)
   kernel(x, out)
   np.testing.assert_array_equal(out, x * 3 + 7)

   reloaded = load_built_artifact(kernel._built_artifact)
   reloaded(x, out)

The output directory contains C++ source, an RV64 ELF executable and an ABI
manifest. Each invocation serializes arguments into temporary files, starts
QEMU, and copies successful results back into the supplied output arrays.
This is a correctness and integration runner, not an on-device performance
benchmark. A compiler or emulator failure raises an exception; unsuccessful
execution does not replace host outputs.

Packed spike accumulation
-------------------------

``ntl.spike_accumulate(current, spikes, weight_low, weight_high)`` computes
``current + sum(spike[i] * weight[i] for i in range(32))`` using the RVNE
``calc_acc_32`` instruction. Operands have these exact types:

* ``current``: ``int32``.
* ``spikes``: ``uint32``, with spike zero in the least significant bit.
* ``weight_low`` and ``weight_high``: ``uint64`` holding sixteen signed INT4
  weights each, with weight zero in the least significant nibble.

Operands must broadcast to the current tensor's shape. This permits shared
spike rows and weight columns without copying them for every output element.
The result has the current tensor's shape and ``int32`` dtype. The helper initializes its
temporary register operands on every call, reads NCR using
``get_ncr_64``, and does not use the SDK's unsupported ``save_nc_32`` operation.
It clobbers its temporary RVNE registers; it does not preserve an external
hardware neuron context.

.. code-block:: python

   import ninetoothed.language as ntl
   from ninetoothed.rvne import pack_spikes, pack_weights

   def accumulate(
       current: Tensor(shape=(1,), dtype="int32"),
       spikes: Tensor(shape=(1,), dtype="uint32"),
       low: Tensor(shape=(1,), dtype="uint64"),
       high: Tensor(shape=(1,), dtype="uint64"),
       out: Tensor(shape=(1,), dtype="int32"),
   ):
       out = ntl.spike_accumulate(current, spikes, low, high)

   kernel = aot(accumulate, backend="rvne", output_dir="build/rvne")
   spikes = np.ones((1, 32), dtype=np.int32)
   weights = np.tile(np.arange(-8, 8, dtype=np.int32), (1, 2))
   packed_weights = pack_weights(weights)
   current = np.zeros(1, dtype=np.int32)
   result = np.zeros_like(current)
   kernel(
       current,
       pack_spikes(spikes)[:, 0].copy(),
       packed_weights[:, 0].copy(),
       packed_weights[:, 1].copy(),
       result,
   )
   np.testing.assert_array_equal(result, [-16])

Packing operates on the last axis, pads incomplete words with zeros, and
rejects nonbinary spikes, floating-point weights and integers outside
``[-8, 7]``. It performs no implicit quantization. Pad weights to a multiple
of 32 when pairing two weight words with each spike word. Larger dot products
can loop over packed words while carrying the returned current. Convolution
patches and dense layers can use the same primitive after explicit layout
conversion; automatic float convolution quantization/fusion is not provided.

Support and validation
----------------------

The ``rvne`` target profile is specific to ``rv64imafcvzne`` and accepts AOT
only. The runner requires NumPy arrays with exact supported dtypes,
C-contiguous storage, compatible source shapes, and writable outputs.
Distinct tensor arguments must not overlap. JIT, jagged tensors, arbitrary
math calls, atomics and unsupported dtypes are rejected. Source generation
does not silently route unsupported kernels to another backend. Signed
arithmetic is compiled with ``-fwrapv``. Shift counts must be supported
compile-time constants.

Supported element types are ``bool``, ``int32``, ``uint32``, ``int64``,
``uint64`` and ``float32``. The initial arithmetic set includes addition,
subtraction, multiplication, comparisons, bitwise operations, selection and
sum/min/max reductions. FP32 ``exp2`` and division are supported; floating
division also accepts an integer scalar operand. Integer division, floor
division and remainder are currently rejected; index calculations have
separate integer floor/modulo handling.

CPU tests validate source generation and ABI handling without the SDK.
Configuring the SDK additionally enables real cross-compilation and QEMU
tests:

.. code-block:: bash

   python -m pytest tests/test_rvne_*.py

These cover ordinary addition, dense matrix multiplication, sum/min/max,
signed INT4 packing, masked tails, accumulated dot products larger than 1024
inputs, explicit LIF state across time steps, blocked attention, failure
handling and artifact reload. QEMU validation does not establish support or
timing on a physical chip. Dedicated LIF instructions, hardware launch/context
management, and register-resident batching remain separate optimization work.

Ordinary operator tests on QEMU
------------------------------

``tests/test_rvne_operators.py`` runs ordinary operators through the complete
NineToothed Python application, SSA, RVNE source generation, SDK compilation
and QEMU execution path. Each case compares the output against NumPy and
checks that compilation produced a RISC-V ELF executable. The tests require
``NINETOOTHED_RVNE_TOOLCHAIN`` and skip if it is unset.

The cases cover ``int32`` and ``float32`` addition, rectangular matrix
multiplication, and both full-vector and row-wise sum/min/max reductions.
Inputs include negative values, zeros and fractional floating-point values.
The matrix shapes include odd dimensions. These operators use ordinary
RISC-V arithmetic; the packed spike accumulator is tested separately.

.. code-block:: bash

   # Run all ordinary operator cases through the real SDK and QEMU.
   python -m pytest -vv -ra tests/test_rvne_operators.py

   # Select one operator family.
   python -m pytest -vv tests/test_rvne_operators.py -k add
   python -m pytest -vv tests/test_rvne_operators.py -k matmul
   python -m pytest -vv tests/test_rvne_operators.py -k reduction

Every selected case must report ``PASSED`` to establish QEMU coverage.
The host-compiled tests in ``test_rvne_lowering.py`` remain useful for checking
source generation on machines without the SDK.

Blocked attention on QEMU
------------------------

``tests/test_rvne_attention.py`` reuses the unchanged arrangement and
application from ``tests/test_attention.py``. It compiles the blocked online
softmax algorithm through the RVNE backend and runs its RISC-V ELF in QEMU.
The kernel retains the key/value block loop and the running maximum,
normalizer and output accumulator. It does not allocate a score tensor
spanning the complete sequence in both dimensions.

The FP32 cases cover causal and noncausal attention, two batches and two heads,
sequence lengths 1, 5 and 7, head dimensions 4 and 64, and query/key block sizes
2-by-4 and 4-by-2. Nondivisible sequence lengths exercise tail masks. The
independent reference is CPU PyTorch ``scaled_dot_product_attention`` with
``scale=1.0`` and ``dropout_p=0.0``, matching the existing NineToothed application.
Results must be finite and agree within ``rtol=2e-5`` and ``atol=2e-5``.

.. code-block:: bash

   python -m pytest -vv -ra tests/test_rvne_attention.py

This verifies the algorithm's FP32 correctness on the tested shapes. The RVNE
emitter uses serial RISC-V floating-point arithmetic and libm ``exp2f``;
it does not provide the GPU memory scheduling or performance of an optimized
GPU FlashAttention kernel. FP16/BF16 tensors remain unsupported by this
backend. Physical-chip performance has not been validated.

SConvLif model verification
--------------------------

``examples/rvne/sconv_lif.py`` runs the supplied SConvLif reference model's
complete four-step graph using NineToothed AOT operators and QEMU. The
external model directory must contain ``yolo_origin.py`` with the supported
``SConvLifOrigin`` model. Its source and weights are not distributed with
NineToothed. This is an adapter for that model, not an ONNX model importer.

The adapter uses the reference fixture's seed ``20260830`` and projects the
initial weights to zero and one, as its ``check_models.py`` specifies. The
input is ``[4,3,64,64]`` and the final output is ``[1,3,2,2,9]``. There is no
trained checkpoint in this fixture, so it measures numerical correctness,
not object-detection accuracy.

.. code-block:: bash

   export NINETOOTHED_RVNE_TOOLCHAIN=/path/to/toolchain
   export NINETOOTHED_SCONVLIF_MODEL_DIR=/path/to/SConvLif
   export PYTHONPATH="$PWD/src"

   python examples/rvne/sconv_lif.py \
     --model-dir "$NINETOOTHED_SCONVLIF_MODEL_DIR" \
     --output-dir build/rvne-sconv-lif

The model executes 40 convolutions, 12 max pools, and 36 LIF updates through
QEMU, for 88 calls. Python handles scheduling, data layout, packing and the
Detect output reshape. Convolutions use the RVNE spike accumulator; pooling
and LIF use generated RISC-V integer arithmetic. The nine LIF layers retain
independent voltage/current state across all four time steps.

Each layer consumes the previous QEMU result. A separate CPU PyTorch model
provides comparison values, never replacement activations. The verifier
compares every convolution, pool, spike output, voltage/current state and
Detect reshape, then the final output: 165 exact comparisons. Success prints:

.. code-block:: text

   PASS: 88 QEMU launches, 165 exact comparisons, 108 nonzero output values.

The output directory contains:

* ``report.json``: pass/fail status, all comparisons, reference source hash,
  compiler identity, launch counts and artifact paths.
* ``reference.npz``: input sequence and reference final output.
* ``qemu_output.npz``: QEMU final output and all final LIF states.
* ``kernels/``: generated C++, RISC-V ELF executables and ABI manifests.

The report is reset to ``running`` at startup and records ``failed`` if
compilation, execution or comparison raises an exception. Successful
completion requires all 88 QEMU calls. The command exits with an error when
validation fails.

Packed weights are reused across time steps. Packed input windows and weights
are stored separately and broadcast inside the generated kernel, avoiding
copies of the weights for every output position. Reported staging bytes
estimate explicit operator buffers and exclude model/reference storage,
serialization temporaries and process RSS. Wall times include compilation,
file transport and simulation, and do not measure physical-chip performance.

To run the corresponding integration test with both environment variables
set:

.. code-block:: bash

   python -m pytest -vv -ra tests/test_rvne_model.py

Without the external model directory, only the full-model case skips. With
the SDK configured, the remaining cases still run real QEMU checks for signed
convolution, non-square kernels, stride/padding, pooling and LIF state.

Native single-executable model
------------------------------

``examples/rvne/sconv_lif_native.py`` builds the same model as one statically
linked RV64 ELF. Python prepares the fixture, generates NineToothed kernels
and checks results. During inference, the target C++ program owns all four
time steps, intermediate buffers and the nine independent voltage/current
states. Packing, pooling gathers and the Detect layout conversion also run
inside that program. One QEMU invocation completes all 88 operator calls.

.. code-block:: bash

   export PYTHONPATH="$PWD/src"
   export NINETOOTHED_RVNE_TOOLCHAIN=/path/to/toolchain
   export NINETOOTHED_SCONVLIF_MODEL_DIR=/path/to/SConvLif

   python -m examples.rvne.sconv_lif_native \
     --model-dir "$NINETOOTHED_SCONVLIF_MODEL_DIR" \
     --output-dir build/rvne-sconv-lif-native

Success requires 164 trace records containing every layer output and LIF
state, plus the final output comparison:

.. code-block:: text

   PASS: one QEMU invocation, 88 native operator calls, 165 exact comparisons.

To generate the executable and fixture without running QEMU, add
``--build-only``. After building, inference can run directly from a shell
without activating Python or importing PyTorch:

.. code-block:: bash

   "$NINETOOTHED_RVNE_TOOLCHAIN/qemu/bin/qemu-riscv64" \
     -L "$NINETOOTHED_RVNE_TOOLCHAIN/xuantie-gnu-toolchain/sysroot" \
     build/rvne-sconv-lif-native/bundle/sconv_lif.elf \
     build/rvne-sconv-lif-native/input.bin \
     build/rvne-sconv-lif-native/output.bin

   cmp build/rvne-sconv-lif-native/output.bin \
       build/rvne-sconv-lif-native/expected.bin

The input and output files contain raw little-endian float32 values, with
shapes ``[4,3,64,64]`` and ``[1,3,2,2,9]``. An optional third filename after
the executable's output argument enables the binary stage trace. Weights
are embedded in the ELF; ``weights.bin`` is a build input and is not needed
when deploying the executable.

The bundle preserves separate kernel translation units and ABI wrappers,
then links them with ``native_runtime.cpp`` and the model driver. Its
``native_manifest.json`` records source hashes, launch ABIs, embedded-weight
hashes, compiler identity and ELF identity. External C files are compiled as
GNU C11, including legacy ``asm`` extensions, and linked with C++17 code.

The model driver exports the original ``main_graph`` C ABI. Defining
``NINETOOTHED_NO_MAIN`` removes its command-line entry so an existing C
``main`` can be linked instead. This driver is a native adaptation of the
supplied graph; it is not a rebuilt output of the original ONNX/MLIR compiler.

Compatibility with the original convolution ABI
----------------------------------------------

``native_runtime.h`` also exports the original square and non-square
``conv_compute_currents_from_f32_ptr_defaultNP_f32out`` entrypoints. They
accept caller-owned NCHW/OIHW float32 buffers and element counts, convert
nonzero input to spikes, round finite weights halfway away from zero and
clip to signed INT4, then call the matching NineToothed kernel. Caller-provided
weights are used on each call. Count errors, non-finite weights and shapes
without a compiled registry entry return an error without changing output.

These adapters preserve the function interface for the shapes compiled into
the bundle; they do not compile new shapes at inference time. Convolution
resets RVNE neuron registers and parameters, matching the old SConv runtime's
defaults. Persistent LIF state must stay in memory, and users must serialize
access to the RVNE register file. The model driver satisfies this contract
with separate input/output state buffers.

.. code-block:: bash

   python -m pytest -vv -ra tests/test_rvne_standalone.py

The tests use a real GNU C caller, SDK compilation and QEMU to check the old
ABI, fractional weights, nonbinary input, malformed arguments, register
configuration left by previous calls, pooling and multi-step LIF. With the
external model directory configured, they also verify the complete native
model in one QEMU invocation and link the unchanged supplied ``main_test.c``
against the new ``main_graph``. The latter runs from a directory containing
only the executable and its input/reference data, with strict output checks.
The original generated graph code can reuse
the convolution ABI, but replacing its inlined pooling/LIF regions still
requires corresponding graph-lowering changes.
