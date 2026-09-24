Ahead-of-Time Kernel Building
=============================

:doc:`The Basics <basics>` tutorial introduced ``ninetoothed.make``, which builds a kernel just-in-time: The first time you call the returned kernel, NineToothed compiles the kernel for the input shapes and dtypes it sees, then caches the result for future calls with matching inputs.

For production deployments, just-in-time compilation has two drawbacks:

- The first call is slow.
- The compiled artifacts live only in the current Python process.

``ninetoothed.build`` compiles configured variants ahead of time and emits native ``.so`` artifacts that Python can load in later runs. Supported Triton builds also export C++ sources and a public header for applications that run without Python.

Basic Usage
-----------

Let's start with the vector addition kernel from the Basics tutorial and port it to ``ninetoothed.build``:

.. code-block:: python

    import functools
    import pathlib

    import ninetoothed
    from ninetoothed import Tensor


    def arrangement(x, y, z, block_size):
        return x.tile((block_size,)), y.tile((block_size,)), z.tile((block_size,))


    def application(x, y, z):
        z = x + y  # noqa: F841


    def premake(block_size):
        arrangement_ = functools.partial(arrangement, block_size=block_size)
        tensors = tuple(Tensor(1, dtype=ninetoothed.float16) for _ in range(3))

        return arrangement_, application, tensors


    configs = tuple(
        ((), {"block_size": block_size}, {}) for block_size in (512, 1024, 2048)
    )

    output_dir = pathlib.Path("./add_build")
    output_dir.mkdir(parents=True, exist_ok=True)

    kernel = ninetoothed.build(
        premake,
        configs,
        meta_parameters=("block_size",),
        kernel_name="add",
        output_dir=output_dir,
        lazy=True,
    )

Compared with ``ninetoothed.make``, the shape of the code is the same—an arrangement function and an application function—but we now wrap the kernel-specific setup in a ``premake`` function and enumerate the variants to compile in ``configs``.

Two practical notes about the call to ``ninetoothed.build``:

- The build creates ``output_dir`` when needed.
- ``lazy=True`` defers compilation to the first Python call. Use ``lazy=False`` when generating native artifacts for a separate C++ build.

We can invoke ``kernel`` like this:

.. code-block:: python

    import torch


    if __name__ == "__main__":
        dtype = torch.float16
        device = "cuda"

        x = torch.tensor((1, 2, 3), dtype=dtype, device=device)
        y = torch.tensor((4, 5, 6), dtype=dtype, device=device)

        z = torch.empty_like(x)
        kernel(x, y, z)

        reference = torch.tensor((5, 7, 9), dtype=dtype, device=device)
        assert torch.allclose(z, reference)

The main guard keeps the example invocation separate from import-time setup.

The ``premake`` Function
------------------------

``premake`` builds everything ``ninetoothed.make`` needs—``arrangement``, ``application``, and the ``tensors``—for a single variant. ``ninetoothed.build`` calls ``premake`` once per variant and compiles each result through the SSA compiler.

The intent is that anything that distinguishes one compiled variant from another (dtypes, concrete shapes, block-size choices, ...) appears as a parameter of ``premake``. Everything else stays hardcoded.

Configurations
--------------

Each entry in ``configs`` is a 3-tuple ``(args, kwargs, compilation_configs)``:

- ``args`` and ``kwargs`` are passed directly to ``premake``.
- ``compilation_configs`` contains compilation-time knobs forwarded to ``ninetoothed.make`` (such as ``num_warps`` and ``num_stages``).

In the example above, each config tells ``premake`` which block size to bake in:

.. code-block:: python

    configs = tuple(
        ((), {"block_size": block_size}, {}) for block_size in (512, 1024, 2048)
    )

``configs`` is typically a Cartesian product over the axes you want to specialize on. For a matrix multiplication kernel you might iterate over dtypes and several block-size triples at once.

Meta-Parameters and Auto-Tuning
-------------------------------

Some ``premake`` parameters do not change what the kernel *computes*—only how fast it runs. ``block_size`` in the example is one: any value in ``(512, 1024, 2048)`` produces a functionally equivalent kernel.

Listing ``block_size`` in ``meta_parameters`` tells ``ninetoothed.build`` to treat it as auto-tunable:

- The Python callable benchmarks candidates using actual runtime inputs and caches the selected candidate.
- The C++ dispatcher selects the first configured candidate for each non-meta configuration. It does not run Python or read an auto-tuning CSV. Both callers pass only the non-meta arguments.

Parameters not listed in ``meta_parameters`` are treated as true compile-time variants: each distinct value produces its own ``.so``, and the dispatcher routes runtime calls to the matching one.

Combining Configuration Axes
----------------------------

Data types, block sizes, and compilation knobs like ``num_warps`` and ``num_stages`` live in different parts of a config tuple. Here's a matrix-multiplication kernel that uses all three at once:

.. code-block:: python

    import functools
    import pathlib

    import ninetoothed
    import ninetoothed.language as ntl
    from ninetoothed import Tensor


    def arrangement(input, other, output, block_size_m, block_size_n, block_size_k):
        output_arranged = output.tile((block_size_m, block_size_n))

        input_arranged = input.tile((block_size_m, block_size_k))
        input_arranged = input_arranged.tile((1, -1))
        input_arranged = input_arranged.expand((-1, output_arranged.shape[1]))
        input_arranged.dtype = input_arranged.dtype.squeeze(0)

        other_arranged = other.tile((block_size_k, block_size_n))
        other_arranged = other_arranged.tile((-1, 1))
        other_arranged = other_arranged.expand((output_arranged.shape[0], -1))
        other_arranged.dtype = other_arranged.dtype.squeeze(1)

        return input_arranged, other_arranged, output_arranged


    def application(input, other, output):
        accumulator = ntl.zeros(output.shape, dtype=ntl.float32)

        for k in range(input.shape[0]):
            accumulator += ntl.dot(input[k], other[k])

        output = accumulator  # noqa: F841


    def premake(m, n, k, dtype, block_size_m, block_size_n, block_size_k):
        arrangement_ = functools.partial(
            arrangement,
            block_size_m=block_size_m,
            block_size_n=block_size_n,
            block_size_k=block_size_k,
        )
        tensors = (
            Tensor(shape=(m, k), dtype=dtype),
            Tensor(shape=(k, n), dtype=dtype),
            Tensor(shape=(m, n), dtype=dtype),
        )

        return arrangement_, application, tensors


    configs = tuple(
        (
            (),
            {
                "m": m,
                "n": n,
                "k": k,
                "dtype": dtype,
                "block_size_m": block_size_m,
                "block_size_n": block_size_n,
                "block_size_k": block_size_k,
            },
            {"num_warps": num_warps, "num_stages": num_stages},
        )
        for m, n, k in ((512, 512, 512),)
        for dtype in (ninetoothed.float16, ninetoothed.bfloat16)
        for block_size_m in (64, 128)
        for block_size_n in (64, 128)
        for block_size_k in (32,)
        for num_warps in (4,)
        for num_stages in (3,)
    )

    output_dir = pathlib.Path("./mm_build")
    output_dir.mkdir(parents=True, exist_ok=True)

    kernel = ninetoothed.build(
        premake,
        configs,
        meta_parameters=("block_size_m", "block_size_n", "block_size_k"),
        kernel_name="mm",
        output_dir=output_dir,
        lazy=True,
    )

Three kinds of variation show up in the ``configs`` comprehension:

- ``dtype`` is a non-meta ``premake`` keyword. Each distinct value produces its own compiled ``.so``, and the generated dispatcher picks the matching one at runtime based on the supplied dtype configuration argument.
- ``block_size_m``, ``block_size_n``, ``block_size_k`` are listed in ``meta_parameters``. For every ``(m, n, k, dtype)`` key, the Python callable can benchmark the meta combinations at runtime; the C++ dispatcher uses the first candidate. The caller never passes the meta arguments.
- ``num_warps`` and ``num_stages`` live in the third slot of each config tuple (``compilation_configs``). These are compilation knobs forwarded as keyword arguments to ``ninetoothed.make``. Each distinct ``(num_warps, num_stages)`` pair is another compile-time variant that participates in auto-tuning just like the meta parameters.

Caching
-------

Compiled libraries and their C++ source bundles are cached by source, IR, ABI,
configuration, target, and toolchain. Repeated builds publish the cached artifacts
to ``output_dir``, including the C++ files required by external builds.

Calling from C++
----------------

For a supported Triton build, ``output_dir`` contains ``ninetoothed.h``,
``<kernel_name>.h``, and ``*.cpp`` files with embedded device code. Compile all
of those C++ files with C++17, the CUDA headers, pthread support, and the CUDA
Driver library. For example:

.. code-block:: bash

    nvcc -std=c++17 -Xcompiler -pthread add_build/*.cpp app.cpp -Iadd_build -lcuda -o app

The generated files can be moved together to another directory. They do not
require the build machine's NineToothed cache, Python, or the generated ``.so``
at runtime. Device code still targets the GPU architecture selected at build time.

The public entry point is ``launch_<kernel_name>``. Its arguments are a
``NineToothedStream`` followed by one ``NineToothedTensor`` per application
argument, then the non-meta configuration arguments in ``premake`` signature
order. The result is a CUDA Driver error code, with zero indicating success.

``NineToothedTensor`` contains ``void *data``, ``uint64_t *shape``, and
``int64_t *strides``. Shapes and strides describe the original tensor, with
strides measured in elements. Tensor data points to device memory; scalar data
points to host storage. Floating-point scalars use ``double`` storage, and
integer scalars use their declared width and signedness. ``NineToothedStream``
is a ``void *`` holding a CUDA stream; the caller must make its CUDA context
current before launching.

Configuration dtype arguments use the constants in ``ninetoothed.h``, such as
``NINETOOTHED_FLOAT32``. Integer, boolean, and ``None`` configuration values use
``int``; floating-point configuration values use ``double``. Empty outputs return
success without launching, and unsupported configuration keys return an error.

``export_cpp=None`` (the default) emits C++ files when every variant uses the
supported Triton tensor ABI and configuration types. Python-only configurations,
such as jagged tensors or arbitrary string keys, remain available through the
returned callable. Set ``export_cpp=True`` to require native exports and receive
an error for an unsupported configuration, or ``export_cpp=False`` to build only
the Python-loadable artifacts.
