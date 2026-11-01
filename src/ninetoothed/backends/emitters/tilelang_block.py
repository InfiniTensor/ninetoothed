"""Demand-driven materialization of pure SSA tiles for TileLang.

Matrix products and reductions are block operations; elementwise expressions
reuse their materialized producers through the common coordinate emitter.
Unsupported control flow, effects, or resources retain the scalar emitter.
"""

from dataclasses import replace

from ninetoothed.backends.emitters import ssa as common
from ninetoothed.backends.emitters.context import EmitContext
from ninetoothed.backends.tilelang_plan import INLINE_OPS as _INLINE
from ninetoothed.backends.tilelang_plan import static_shape as _shape


class _Scope:
    def __init__(self, emitter, block, parent=None):
        self.emitter = emitter
        self.block = block
        self.parent = parent
        self.lines = []
        self.buffers = {}
        self.bindings = {}
        self.names = {value.name for value in block.args}
        self.names.update(value.name for op in block.operations for value in op.results)

    def inherited(self, attribute):
        result = self.parent.inherited(attribute) if self.parent else {}
        result.update(getattr(self, attribute))

        return result

    def context(self, shape=(), coords=(), *, scalar=True, accessed=None):
        module = self.emitter.module
        target = replace(
            self.emitter.target,
            tile_values=self.inherited("buffers"),
            scalar_global_access=scalar,
            _accessed_buffers=accessed,
        )
        linear = common.linearized_index(coords, tuple(map(str, shape)))
        ctx = EmitContext(
            target=target,
            kernel=module.kernel,
            program=module.kernel.ssa,
            operations=module.operations,
            value_types=module.value_types,
            lines=[],
            memo={},
            tensor_infos=module.tensors,
            output=module.stores[0].operands[1],
            output_axes=tuple(map(str, shape)),
            index_expr=common.linearized_index(coords, tuple(map(str, shape))),
            outer_index_expr="nt_block",
            inner_index_expr=linear,
            coordinate_exprs=coords,
            mask_expr=None,
            bindings=self.inherited("bindings"),
            temp_counter=[0],
            materialized={},
        )
        # Scalar consumers (including store and loop bounds) use emit_value;
        # coordinate consumers use the target hook above. Both read this scope.
        ctx.memo.update(
            {
                name: target.materialized_element(
                    name,
                    common.current_coords(
                        tuple(map(str, module.value_types[name].shape)), ctx
                    ),
                )
                for name in target.tile_values
            }
        )

        return ctx

    def element(self, name, coords):
        ctx = self.context(_shape(self.emitter.module.value_types[name]), coords)
        value = common.emit_element(name, coords, ctx)

        return ctx.lines, value

    def allocate(self, name, type_):
        buffer = self.emitter.fresh()
        shape = _shape(type_) or (1,)
        dtype = "int64" if type_.dtype == "index" else type_.dtype
        self.lines.append(f'{buffer} = T.alloc_fragment({shape!r}, "{dtype}")')

        return buffer

    def fill(self, buffer, name, type_):
        shape = _shape(type_)

        def build(ctx):
            coords = ctx.coordinate_exprs
            value = common.emit_element(name, coords, ctx)
            ctx.lines.append(f"{buffer}[{', '.join(coords) or '0'}] = {value}")

        self.parallel(shape, build)

    def parallel(self, shape, build):
        """Vectorize only proven ordinary accesses on aligned base pointers."""
        coords = tuple(f"nt_c{i}" for i in range(len(shape)))
        accessed = set()
        native = self.context(shape, coords, scalar=False, accessed=accessed)
        build(native)

        def loop(lines):
            if not shape:
                return lines
            return [
                f"for {', '.join(coords)} in T.Parallel({', '.join(map(str, shape))}):",
                *_indent(lines),
            ]

        if not accessed:
            self.lines.extend(loop(native.lines))

            return

        scalar = self.context(shape, coords)
        build(scalar)

        if not all(self.aligned_rows(buffer) for buffer in accessed):
            self.lines.extend(loop(scalar.lines))

            return

        alignment = " & ".join(
            f'T.call_extern("bool", "_nt_aligned", {buffer}.data)'
            for buffer in sorted(accessed)
        )
        self.lines.append(f"if {alignment}:")
        self.lines.extend(_indent(loop(native.lines)))
        self.lines.append("else:")
        self.lines.extend(_indent(loop(scalar.lines)))

    def aligned_rows(self, buffer):
        info = self.emitter.module.tensors[buffer.removesuffix("_buf")]
        attrs = info.attrs or {}

        if (
            attrs.get("layout_operations") is None
            or set(attrs["layout_operations"])
            - {"tile", "expand", "squeeze", "unsqueeze"}
            or attrs.get("custom_tiling", True)
        ):
            return False

        nbytes = {
            "float16": 2,
            "bfloat16": 2,
            "float32": 4,
            "int32": 4,
            "int64": 8,
        }.get(info.dtype)

        try:
            strides = tuple(int(stride) for stride in common.source_strides(info))

            return bool(
                nbytes
                and strides
                and strides[-1] == 1
                and all(stride * nbytes % 64 == 0 for stride in strides[:-1])
            )
        except (TypeError, ValueError):
            return False

    def ensure(self, name):
        if name in self.inherited("buffers") or name in self.inherited("bindings"):
            return

        if self.parent and name not in self.names:
            self.parent.ensure(name)

            return

        op = self.emitter.module.operations.get(name)

        if (
            op is None
            or op.opcode.startswith("dtype.")
            or op.opcode in {"shape.dim", "index.offset"}
        ):
            return

        if op.opcode == "scf.for":
            self.loop(op)

            return

        if op.opcode == "scf.if":
            self.branch(op)

            return

        for operand in op.operands:
            self.ensure(operand)

        if op.opcode in _INLINE:
            return

        type_ = self.emitter.module.value_types[name]

        if op.opcode in {"linalg.dot", "linalg.matmul"}:
            self.dot(op)
        elif op.opcode.startswith("reduce."):
            self.reduce(op)
        elif type_.shape and type_.dtype != "bool":
            buffer = self.allocate(name, type_)
            self.fill(buffer, name, type_)
            self.buffers[name] = buffer

    def dot(self, op):
        staged = []
        transposed = []

        for operand in op.operands[:2]:
            producer = self.emitter.module.operations.get(operand)
            transpose = producer is not None and producer.opcode == "linalg.transpose"

            if transpose:
                operand = producer.operands[0]

            transposed.append(transpose)
            type_ = self.emitter.module.value_types[operand]
            buffer = self.emitter.fresh()
            self.lines.append(
                f'{buffer} = T.alloc_shared({_shape(type_)!r}, "{type_.dtype}")'
            )
            self.fill(buffer, operand, type_)
            staged.append(buffer)

        result = op.results[0]
        buffer = self.allocate(result.name, result.type)
        self.lines.extend(
            (
                f"T.clear({buffer})",
                f"T.gemm({staged[0]}, {staged[1]}, {buffer}, transpose_A={transposed[0]}, transpose_B={transposed[1]}, policy=T.GemmWarpPolicy.FullRow)",
            )
        )
        self.buffers[result.name] = buffer

    def reduce(self, op):
        operand = op.operands[0]
        type_ = self.emitter.module.value_types[operand]
        staged = self.allocate(operand, type_)
        shape = _shape(type_)

        def build(ctx):
            coords = ctx.coordinate_exprs
            term = common.emit_element(operand, coords, ctx)
            mask = common.reduction_input_mask(operand, coords, ctx)

            if mask:
                identity = common.reduction_identity(
                    op.opcode[7:], op.results[0].type, ctx.target
                )
                term = ctx.target.where(mask, term, identity)

            ctx.lines.append(f"{staged}[{', '.join(coords)}] = {term}")

        self.parallel(shape, build)
        result = op.results[0]
        buffer = self.allocate(result.name, result.type)
        self.lines.append(
            f"T.reduce_{op.opcode[7:]}({staged}, {buffer}, dim={len(shape) - 1})"
        )
        self.buffers[result.name] = buffer

    def loop(self, op):
        for attr, result in zip(op.attrs["iter_args"], op.results):
            initial = attr["initial"]
            self.ensure(initial)
            buffer = self.allocate(result.name, result.type)
            self.fill(buffer, initial, result.type)
            self.buffers[result.name] = buffer

        bounds = []

        for operand in op.operands[:3]:
            self.ensure(operand)
            ctx = self.context()
            bounds.append(common.emit_loop_bound(operand, ctx))
            self.lines.extend(ctx.lines)

        child = _Scope(self.emitter, op.regions[0], self)
        induction = self.emitter.fresh()
        child.bindings[op.attrs["induction"]] = induction

        for attr, result in zip(op.attrs["iter_args"], op.results):
            child.buffers[attr["block_arg"]] = self.buffers[result.name]

        yields = op.regions[0].operations[-1].operands
        next_values = []

        for yielded, result in zip(yields, op.results):
            child.ensure(yielded)
            buffer = child.allocate(yielded, result.type)
            child.fill(buffer, yielded, result.type)
            next_values.append((buffer, result))

        for buffer, result in next_values:
            child.lines.append(f"T.copy({buffer}, {self.buffers[result.name]})")

        self.lines.append(self.emitter.target.loop_header(induction, *bounds))
        self.lines.extend(_indent(child.lines))

    def branch(self, op):
        for result in op.results:
            self.buffers[result.name] = self.allocate(result.name, result.type)

        self.ensure(op.operands[0])
        lines, condition = self.element(op.operands[0], ())
        self.lines.extend(lines)
        # Emit branch-independent definitions before the conditional header.
        branches = []

        for region in op.regions:
            child = _Scope(self.emitter, region, self)

            for yielded, result in zip(region.operations[-1].operands, op.results):
                child.ensure(yielded)
                child.fill(self.buffers[result.name], yielded, result.type)

            branches.append(child.lines)

        for i, lines in enumerate(branches):
            self.lines.append(f"if {condition}:" if i == 0 else "else:")
            self.lines.extend(_indent(lines))


class _Emitter:
    def __init__(self, module, target):
        self.module, self.target = module, target
        self.counter = 0

    def fresh(self):
        self.counter += 1

        return f"nt_tile_{self.counter}"


def _indent(lines):
    return common.indent_block("\n".join(lines), "    ").splitlines()


def render(module, target):
    from ninetoothed.backends.emitters.tilelang import (
        _render_tilelang_module_skeleton,
        _tilelang_flat_buffers,
        _tilelang_signature,
    )

    emitter = _Emitter(module, target)
    scope = _Scope(emitter, module.kernel.ssa.blocks[0])
    store = module.stores[0]
    scope.ensure(store.operands[0])
    shape = _shape(module.value_types[store.operands[0]])

    def write(ctx):
        # Store uses the value API; compute inline views at these explicit
        # coordinates first, just as fragment fills do, then reuse that value.
        ctx.memo[store.operands[0]] = common.emit_element(
            store.operands[0], ctx.coordinate_exprs, ctx
        )
        common.emit_operation(store, ctx)

    scope.parallel(shape, write)
    scope.lines.insert(0, "T.import_source(_NT_SCALAR_ACCESS)")
    plan = module.scheduled_metadata["tilelang_emission"]
    body = f"        with T.Kernel({plan['grid']}, threads={plan['threads']}) as nt_block:\n"
    body += common.indent_block("\n".join(scope.lines), "            ")
    _, buffers, signature = _tilelang_signature(
        module.variables,
        module.outputs,
        module.shape_params,
        module.tensors,
        module.value_types,
    )

    return _render_tilelang_module_skeleton(
        module.kernel,
        target,
        note="materialized SSA tiles",
        handle_args=signature,
        buffers=_tilelang_flat_buffers(buffers, module.tensors, module.grid_total),
        body=body,
    )
