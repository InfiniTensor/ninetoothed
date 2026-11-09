RVNE Backend
============

The ``rvne`` backend emits C++ for the RISC-V Zne extension and compiles it
with the RVNE SDK. AOT execution runs through the SDK's user-mode QEMU using
NumPy host arrays. Source generation does not need the SDK.

The current backend provides scalar RV64 arithmetic, control flow and
reductions. A ``bool @ int4`` contraction is pattern-matched and lowered to
the RVNE ``calc_acc_32`` instruction. Integer LIF updates are
currently expressed with ordinary arithmetic, shifts and ``ntl.where``; they
do not yet use the dedicated hardware LIF instruction.

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

Use the historical NineToothed AOT entrypoint. Save the application in a
Python file so the compiler can inspect its source:

.. code-block:: python

   import numpy as np
   from ninetoothed import Tensor
   from ninetoothed.aot import aot

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

The returned handle is the normal execution API. The output directory contains
C++ source, an RV64 ELF executable and an ABI manifest. Each invocation
serializes arguments into temporary files, starts QEMU, and copies successful
results back into the supplied output arrays. This is a correctness and
integration runner, not an on-device performance benchmark. A compiler or
emulator failure raises an exception; unsuccessful execution does not replace
host outputs.

Quantized spike contraction
---------------------------

Applications use ordinary matrix multiplication for an explicitly quantized
spike contraction: the spike operand has dtype ``bool`` and the weight operand
has dtype ``int4``. The frontend infers an ``int32`` result, and the RVNE
emitter pattern-matches this contraction, packs each 32-lane group at runtime,
and calls ``calc_acc_32``. Nonintegral and implicitly quantized floating-point
contractions are not lowered to this path.

.. code-block:: python

   import numpy as np
   from ninetoothed import Tensor
   from ninetoothed.aot import aot

   def accumulate(
       current: Tensor(shape=(2, 3), dtype="int32"),
       spikes: Tensor(shape=(2, 4), dtype="bool"),
       weights: Tensor(shape=(4, 3), dtype="int4"),
       out: Tensor(shape=(2, 3), dtype="int32"),
   ):
       out = current + (spikes @ weights)

   kernel = aot(accumulate, backend="rvne", output_dir="build/rvne")
   current = np.zeros((2, 3), dtype=np.int32)
   spikes = np.array(
       [[0, 1, 0, 1], [1, 0, 1, 0]], dtype=bool
   )
   weights = np.arange(-8, 4, dtype=np.int8).reshape(4, 3)
   out = np.empty_like(current)
   kernel(current, spikes, weights, out)
   np.testing.assert_array_equal(
       out, current + (spikes.astype(np.int32) @ weights.astype(np.int32))
   )

NumPy has no native signed-INT4 storage, so RVNE represents an ``int4`` kernel
argument with native ``int8_t`` bytes. Values must already be integers in
``[-8, 7]``; the runtime rejects anything outside that range instead of
rounding, clipping, or silently truncating it. Reduction extents that are not a
multiple of 32 are padded with zero spikes and zero weights inside the generated
kernel. Convolution patches can use the same lowering after they are explicitly
gathered into a ``bool @ int4`` matrix multiplication; no frontend accumulator
intrinsic is exposed.

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

Supported general element types are ``bool``, ``int32``, ``uint32``,
``int64``, ``uint64`` and ``float32``. ``int4`` is accepted as an explicitly
quantized weight operand for the ``bool @ int4`` contraction path and is
carried over the ABI as native ``int8_t`` bytes. The initial arithmetic set includes addition,
subtraction, multiplication, comparisons, bitwise operations, selection and
sum/min/max reductions. FP32 ``exp2`` and division are supported; floating
division also accepts an integer scalar operand. Integer division, floor
division and remainder are currently rejected; index calculations use
helpers that preserve Python floor division and modulo semantics.

CPU tests validate source generation and ABI handling without the SDK.
Configuring the SDK additionally enables real cross-compilation and QEMU
tests:

.. code-block:: bash

   python -m pytest tests/test_rvne_*.py

These cover ordinary addition, ordinary dense matrix multiplication,
``bool @ int4`` spike contraction, non-32 reduction tails, contractions longer
than 1024 inputs, explicit LIF state across time steps, blocked attention,
failure handling and artifact reload. QEMU validation does not establish support or
timing on a physical chip. Dedicated LIF instructions, hardware launch and
context management, and register-resident batching remain optimization work.

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
