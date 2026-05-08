import pathlib


class CudaAotBackend:
    """CUDA AOT backend.

    This backend keeps the legacy CUDA AOT behavior while allowing
    `aot.py` to focus on backend routing.
    """

    def __init__(self, compile_variants, load_dispatcher):
        self._compile_variants = compile_variants
        self._load_dispatcher = load_dispatcher

    def __call__(
        self,
        func,
        *,
        caller,
        kernel_name,
        output_dir,
        num_warps,
        num_stages,
    ):
        output_dir = pathlib.Path(output_dir)
        output_contents = self._compile_variants(
            func, caller, kernel_name, num_warps, num_stages
        )

        for output_name, output_content in output_contents.items():
            output_path = output_dir / output_name
            with open(output_path, "w") as f:
                f.write(output_content)

        return self._load_dispatcher(kernel_name=kernel_name, output_dir=output_dir)
