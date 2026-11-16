"""One shared domain, def-use and storage plan for native vector targets."""

from dataclasses import dataclass

from ninetoothed.backends.emitters.expressions import c_integer_expr
from ninetoothed.backends.emitters.memory_guard import dense_staging_memory_guard
from ninetoothed.backends.emitters.vector_dag import plan_vector_dag
from ninetoothed.backends.emitters.vector_domain import prove_flat_domain
from ninetoothed.compiler.vector_resources import plan_vector_storage


@dataclass(frozen=True)
class NativeVectorPlan:
    dag: object
    extent: str
    guard: str
    storage: object

    @property
    def tasks(self):
        chunk = self.storage.elements

        return f"(({self.extent}) / {chunk} + (({self.extent}) % {chunk} != 0))"


def plan_native_vector(
    context, classify, *, elements, dtype_bytes, budget, alignment=128
):
    """Targets supply primitive legality and physical limits, never a name switch."""
    extent = prove_flat_domain(context)

    if extent is None:
        return None

    dag = plan_vector_dag(
        context,
        classify,
        supported_dtypes=frozenset(dtype_bytes),
        storage_dtype=lambda dtype: "float32" if dtype == "bool" else dtype,
    )

    if dag is None or any(value.dtype not in dtype_bytes for _, _, value in dag.stores):
        return None

    storage = plan_vector_storage(
        (dtype_bytes[d] for d in dag.buffer_dtypes),
        elements=elements,
        budget=budget,
        alignment=alignment,
    )
    guard = c_integer_expr(
        dense_staging_memory_guard(context, context.variables, context.outputs)
    )

    if guard == "false":
        return None

    return NativeVectorPlan(dag, extent, guard, storage)
