"""Block-local expression reuse with explicit memory and region boundaries."""

from dataclasses import replace

from ninetoothed.compiler.effects import pure_expression
from ninetoothed.compiler.ssa_values import rewrite_references, rewrite_value
from ninetoothed.compiler.ssa_values import structural_key as _key


def eliminate_common_subexpressions(program):
    """Reuse identical typed expressions without reassociation or hoisting."""
    aliases = {}
    eliminated = []

    def rewrite(value):
        return rewrite_references(value, aliases)

    def block_rewrite(block):
        available = {}
        operations = []

        for operation in block.operations:
            current = replace(
                operation,
                operands=tuple(rewrite(name) for name in operation.operands),
                attrs=rewrite(operation.attrs),
                results=tuple(
                    rewrite_value(value, aliases) for value in operation.results
                ),
                regions=tuple(block_rewrite(region) for region in operation.regions),
            )

            if (
                not pure_expression(current)
                and current.opcode
                not in {"linalg.dot", "linalg.matmul", "linalg.transpose"}
            ) or len(current.results) != 1:
                available.clear()
                operations.append(current)
                continue

            result = current.results[0]
            key = (
                current.opcode,
                current.operands,
                _key(current.attrs),
                _key(result.type),
            )

            if key in available:
                aliases[result.name] = available[key]
                eliminated.append((result.name, available[key]))
                continue

            available[key] = result.name
            operations.append(current)

        return replace(
            block,
            args=tuple(rewrite_value(value, aliases) for value in block.args),
            operations=tuple(operations),
        )

    blocks = tuple(block_rewrite(block) for block in program.blocks)

    return (
        replace(
            program,
            outputs=tuple(
                rewrite_value(value, aliases, rename=True) for value in program.outputs
            ),
            blocks=blocks,
        ),
        tuple(eliminated),
    )


__all__ = ["eliminate_common_subexpressions"]
