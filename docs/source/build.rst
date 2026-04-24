Ahead-of-Time Kernel Building
=============================

:doc:`The Basics <basics>` tutorial introduced ``ninetoothed.make``, which builds a kernel just-in-time: The first time you call the returned kernel, NineToothed compiles the kernel for the input shapes and dtypes it sees, then caches the result for future calls with matching inputs.

For production deployments, just-in-time compilation has two drawbacks:

- The first call is slow.
- The compiled artifacts live only in the current Python process.

``ninetoothed.build`` addresses both by compiling kernels **ahead of time**, emitting native ``.so`` artifacts to disk that can be loaded directly in later runs.

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

- ``output_dir`` must exist. ``ninetoothed.build`` writes generated sources into it but does not create it.
- ``lazy=True`` defers the actual build to the first kernel call. This is required when ``ninetoothed.build`` is invoked at module import time, because the ``ProcessPoolExecutor`` that drives the build uses ``spawn`` and cannot re-enter the importing module.

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

The ``if __name__ == "__main__":`` guard is also required: the first call to ``kernel`` is what actually drives the build, and the build's child processes re-import the enclosing module. Without the guard, the children would try to call the kernel again on import and deadlock.

The ``premake`` Function
------------------------

``premake`` builds everything ``ninetoothed.make`` needs—``arrangement``, ``application``, and the ``tensors``—for a single variant. ``ninetoothed.build`` calls ``premake`` once per variant and hands each result to ``ninetoothed.make`` under the hood.

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

- At build time, ``ninetoothed.build`` benchmarks each meta variant against representative inputs and records the winner per non-meta configuration in a CSV file next to the ``.so``.
- At runtime, the generated C++ dispatcher picks the best meta values from the CSV based on the actual input arguments, so the caller passes only the non-meta arguments.

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

- ``dtype`` is a non-meta ``premake`` keyword. Each distinct value produces its own compiled ``.so``, and the generated dispatcher picks the matching one at runtime based on the tensor's dtype.
- ``block_size_m``, ``block_size_n``, ``block_size_k`` are listed in ``meta_parameters``. For every ``(m, n, k, dtype)`` key, ``ninetoothed.build`` benchmarks the meta combinations at build time and records the winner in the CSV; the caller never passes these at runtime.
- ``num_warps`` and ``num_stages`` live in the third slot of each config tuple (``compilation_configs``). These are compilation knobs forwarded as keyword arguments to ``ninetoothed.make``. Each distinct ``(num_warps, num_stages)`` pair is another compile-time variant that participates in auto-tuning just like the meta parameters.

Caching
-------

``ninetoothed.build`` writes all generated artifacts—generated C++ sources, ``.so``, and the auto-tuning CSV—into ``output_dir``. On a later invocation with the same ``output_dir``, if the ``.so`` already exists and the CSV has entries, ``ninetoothed.build`` loads them directly and skips the full build. This makes repeated imports of a ``build``-based kernel effectively free after the first run.

If you want to force a rebuild, delete ``output_dir``.

Sizing the Warm-Up Tensors
--------------------------

Auto-tuning needs concrete tensor shapes, but the tensor specs returned by ``premake`` may have symbolic dimensions. For each symbolic dimension, ``ninetoothed.build`` defaults to a preset size when materializing warm-up tensors.

For kernels with large fixed dimensions (such as a matmul with ``K`` or ``N`` in the hundreds of thousands), you can cap a particular symbolic dimension's warm-up size by setting ``upper_bound`` in ``shape_options``:

.. code-block:: python

    def premake(k, n, dtype):
        shape_options = ({"upper_bound": 4}, None, None)
        tensors = (
            Tensor(shape=(None, None, k), shape_options=shape_options, dtype=dtype),
            Tensor(shape=(None, k, n), shape_options=shape_options, dtype=dtype),
            Tensor(shape=(None, None, n), shape_options=shape_options, dtype=dtype),
        )

        return arrangement, application, tensors

The ``upper_bound`` is a warm-up hint only—it does not constrain the kernel at runtime. Useful when a symbolic dimension is known to stay small in practice (for example, the batch dimension of a batched matmul) while other symbolic dimensions remain large.
