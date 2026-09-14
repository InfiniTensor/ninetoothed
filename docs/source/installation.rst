Installation
============

You can install NineToothed using ``pip``:

.. code-block::

    pip install ninetoothed

To fully leverage its capabilities, you will also need to install a compatible deep learning framework. Currently, NineToothed supports `PyTorch <https://pytorch.org/>`_.

It is generally considered good practice to use a virtual environment when installing packages with pip, though it is optional. You may find this `documentation <https://docs.python.org/3/library/venv.html>`_ helpful.

.. _cpu-wheel-install:

CPU interpreter without GPU packages
------------------------------------

The NumPy reference interpreter and its step debugger can run without Torch,
Triton, or a GPU. Use this checkout containing ``ninetoothed.interpret``; an
unrelated PyPI release with the same version number may not contain it.

The following Bash commands start in the repository root, require Python 3.10+
with ``venv`` and pip, and create a fresh environment outside the checkout.
They build the normal wheel, install its NumPy/SymPy runtime requirements, and
explicitly skip its GPU dependency. No editable install or ``PYTHONPATH`` is
needed. Build and dependency installation require access to a package index
or a populated local cache.

.. code-block:: bash

   REPO=$PWD
   CPU_WORK=$(mktemp -d)
   python3 -m venv "$CPU_WORK/venv"
   source "$CPU_WORK/venv/bin/activate"
   python -m pip wheel --no-deps --wheel-dir "$CPU_WORK/wheels" .
   python -m pip install "numpy>=1.26.4" "sympy>=1.13.0"
   python -m pip install --no-deps "$CPU_WORK"/wheels/ninetoothed-*.whl

   # Copy the example, then run outside the source checkout.
   cp docs/cpu_interpreter_demo.py "$CPU_WORK/demo.py"
   cd "$CPU_WORK"
   python -I -c 'import importlib.util, ninetoothed; print(ninetoothed.__file__); assert all(importlib.util.find_spec(p) is None for p in ("torch", "triton"))'
   python -I demo.py --debug --export replay
   python -I replay/replay.py

If your Python distribution lacks ``ensurepip``, install its OS ``venv`` support,
or use an existing ``uv`` installation to replace the environment-creation line
with ``uv venv --seed --python python3 "$CPU_WORK/venv"``. Then continue with
activation and the same pip commands. The isolated Linux validation used this
``uv`` alternative because its system Python lacked ``ensurepip``.

The printed package path should be inside ``venv/lib/python*/site-packages``.
``-I`` ignores ``PYTHONPATH`` and excludes the current/script directory from
the import search path. The demo verifies ``x * 2 + 1`` against NumPy, exercises
stepping and breakpoints, then locates a deliberately injected constant error.
The saved replay verifies the reference and reproduces that error independently.
See :doc:`cpu_interpreter` for the interpreter's supported semantics and limits.

This is an explicit CPU installation override, not a separate CPU distribution:
the wheel's metadata still declares ``triton>=3.0.0``. Consequently,
``python -m pip check`` reports missing Triton. Do not treat that environment as
dependency-complete for GPU compilation. A later normal install, upgrade, or
``[all]``/``[debugging]`` extra can install GPU packages again. The step debugger
used above is part of the NumPy interpreter and does not need the older
Torch-based ``[debugging]`` extra.

To run the interpreter's CPU regressions against the installed wheel:

.. code-block:: bash

   python -m pip install pytest
   cp -r "$REPO/tests" "$CPU_WORK/tests"
   env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
       python -m pytest --import-mode=importlib -q \
       tests/test_interpreter_applications.py \
       tests/test_interpreter_ssa.py \
       tests/test_interpreter_debugger.py \
       tests/test_interpreter_step_debugger.py \
       tests/test_interpreter_matmul.py \
       tests/test_interpreter_provenance.py \
       tests/test_interpreter_default_pipeline.py

This test copy contains no ``src`` directory. Some tests import their helper
functions in child processes, so run pytest from ``CPU_WORK`` as shown, without
``-I``. The demo and standalone replay above can use ``-I``.

The full repository suite includes GPU/framework tests and is not a CPU-only
installation check. The commands above do not measure GPU performance.
