from tests.utils import skip_if_cuda_not_available


@skip_if_cuda_not_available
def test_ipynb():
    import inspect
    import os
    import subprocess
    import textwrap

    def _test_sin():
        import torch

        import ninetoothed
        from ninetoothed import Tensor, block_size
        from ninetoothed.language import libdevice

        def arrangement(input, output, BLOCK_SIZE=block_size()):
            return (input.tile((BLOCK_SIZE,)), output.tile((BLOCK_SIZE,)))

        def application(input, output):
            output = libdevice.sin(input)  # noqa: F841

        def sin(input):
            output = torch.empty_like(input)

            sin_kernel = ninetoothed.make(
                arrangement, application, (Tensor(1), Tensor(1))
            )

            sin_kernel(input, output)

            return output

        size = 51252
        device = "cuda"

        input = torch.randn(size, device=device)

        assert torch.allclose(sin(input), torch.sin(input))

    PARENT_DIR_PATH = os.path.dirname(__file__)
    PY_FILE_NAME = "test_sin.py"
    PY_FILE_PATH = os.path.join(PARENT_DIR_PATH, PY_FILE_NAME)
    IPYNB_FILE_NAME = os.path.splitext(PY_FILE_NAME)[0] + ".ipynb"
    IPYNB_FILE_PATH = os.path.join(PARENT_DIR_PATH, IPYNB_FILE_NAME)
    NBCONVERT_IPYNB_FILE_NAME = os.path.splitext(PY_FILE_NAME)[0] + ".nbconvert.ipynb"
    NBCONVERT_IPYNB_FILE_PATH = os.path.join(PARENT_DIR_PATH, NBCONVERT_IPYNB_FILE_NAME)

    with open(PY_FILE_PATH, "w") as f:
        f.write(textwrap.dedent("".join(inspect.getsourcelines(_test_sin)[0][1:])))

    subprocess.run(("jupytext", "--to", "ipynb", PY_FILE_PATH), check=True)
    subprocess.run(
        ("jupyter", "nbconvert", "--to", "notebook", "--execute", IPYNB_FILE_PATH),
        check=True,
    )

    os.remove(PY_FILE_PATH)
    os.remove(IPYNB_FILE_PATH)
    os.remove(NBCONVERT_IPYNB_FILE_PATH)
