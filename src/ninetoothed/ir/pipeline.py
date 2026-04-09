import ast
import inspect
import pathlib
import textwrap

from ninetoothed.generation.generation import cache_source
from ninetoothed.generation.node_transformer_utils import _Inliner
from ninetoothed.ir.passes.ast_to_l1 import ASTToL1Pass
from ninetoothed.ir.passes.l1_to_l2 import L1ToL2Pass
from ninetoothed.ir.passes.l2_to_l3 import L2ToL3Pass
from ninetoothed.ir.passes.l3_to_code import L3ToCodePass


class IRPipeline:

    def __init__(self, context, args, caller, kernel_name, num_warps, num_stages,
                 max_num_configs, prettify=False, dump_ir=False):
        self._context = context
        self._args = args
        self._caller = caller
        self._kernel_name = kernel_name
        self._num_warps = num_warps
        self._num_stages = num_stages
        self._max_num_configs = max_num_configs
        self._prettify = prettify
        self._dump_ir = dump_ir

        # Populated during run()
        self.l1_func = None
        self.l2_func = None
        self.l3_func = None
        self.l4_source = None

    def run(self, func):
        """Run the IR pipeline on a Python function.

        :param func: The Python kernel function
        :return: Path to the cached source file
        """

        # L0: Parse AST
        tree = self._get_tree(func)

        # L1: Lower To TensorTile IR
        l1_pass = ASTToL1Pass(self._context)
        self.l1_func = l1_pass.transform(tree)

        if self._dump_ir:
            print("=" * 80)
            print("L1: Tensor Tile IR")
            print("=" * 80)
            print(self.dump_l1())

        exit(0)

        # L2: Lower To Memory IR
        l2_pass = L1ToL2Pass()
        self.l2_func = l2_pass.transform(self.l1_func)

        if self._dump_ir:
            print()
            print("=" * 80)
            print("L2: Memory IR")
            print("=" * 80)
            print(self.dump_l2())

        # L2 → L3: Generate Triton IR
        l3_pass = L2ToL3Pass()
        self.l3_func = l3_pass.transform(self.l2_func)

        if self._dump_ir:
            print()
            print("=" * 80)
            print("L3: Triton IR")
            print("=" * 80)
            print(self.dump_l3())

        # L3 → L4: Generate Triton Python code
        l4_pass = L3ToCodePass(prettify=self._prettify)
        self.l4_source = l4_pass.transform(self.l3_func)

        if self._dump_ir:
            print()
            print("=" * 80)
            print("Generated Triton Python Code")
            print("=" * 80)
            print(self.dump_l4())

        # Cache the generated source
        cache_file = cache_source(self.l4_source)

        return cache_file

    def dump_l1(self, indent=0):
        """Dump L1 Tensor+Tiling IR in MLIR-style SSA format."""
        if self.l1_func is not None:
            return self.l1_func.dump_ssa()
        return "<L1 not yet generated>"

    def dump_l2(self, indent=0):
        """Dump L2 Memory IR in MLIR-style SSA format."""
        if self.l2_func is not None:
            return self.l2_func.dump_ssa()
        return "<L2 not yet generated>"

    def dump_l3(self, indent=0):
        """Dump L3 Triton IR in MLIR-style SSA format."""
        if self.l3_func is not None:
            return self.l3_func.dump_ssa()
        return "<L3 not yet generated>"

    def dump_l4(self):
        """Dump L4 generated code."""
        if self.l4_source is not None:
            return self.l4_source
        return "<L4 not yet generated>"

    def _get_tree(self, func):
        """Parse function source to AST with inlining.

        Reuses the same logic as CodeGenerator._get_tree.
        """
        func_def = ast.parse(textwrap.dedent(inspect.getsource(func)))

        inliner = _Inliner(func.__globals__)
        inliner.visit(func_def)

        if inliner.libdevice_used:
            libdevice_alias = ast.alias(
                name="libdevice", asname=_Inliner.LIBDEVICE_ALIAS
            )
            libdevice_import = ast.ImportFrom(
                module="triton.language.extra",
                names=[libdevice_alias],
                level=0,
            )
            func_def.body.insert(0, libdevice_import)

        return func_def
