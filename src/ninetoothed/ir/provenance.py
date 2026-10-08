"""Explicit SSA transformation provenance, without claiming value equivalence.

Origins identify operations in the seeded SSA, not Python source lines. A pass
must declare replacement, split, merge and deletion relations. Object identity
is sufficient only for operations that were actually retained. Similar names,
opcodes and structure never establish a relation. An origin shared by several
generated operations describes a source candidate, not a unique fault location.
"""

import hashlib
from dataclasses import dataclass, replace

from . import ssa


def operation_locations(program):
    """Return static locations using the interpreter's nested-region convention."""

    def visit(block, path):
        for index, operation in enumerate(block.operations):
            location = f"{path}:{index}:{operation.opcode}"
            yield location, operation

            for region_index, region in enumerate(operation.regions):
                yield from visit(region, f"{location}/region{region_index}")

    if len(program.blocks) != 1:
        raise ValueError("Provenance requires exactly one SSA entry block.")
    return tuple(visit(program.blocks[0], "entry"))


def _fingerprint(program):
    return hashlib.sha256(ssa.render(program).encode("utf-8")).hexdigest()


def _metadata(program):
    data = program.metadata.get("provenance")

    if data is None or data.get("schema") != 1:
        raise ValueError("Seed SSA origins before recording a pass.")

    sources = set(data["sources"])

    for _location, operation in operation_locations(program):
        if any(origin not in sources for origin in operation.origins):
            raise ValueError("Operation refers to an unknown SSA origin.")
    return data


def _map_locations(program, transform):
    def block(value, path):
        operations = []

        for index, operation in enumerate(value.operations):
            location = f"{path}:{index}:{operation.opcode}"
            regions = tuple(
                block(region, f"{location}/region{region_index}")
                for region_index, region in enumerate(operation.regions)
            )
            operations.append(replace(transform(location, operation), regions=regions))
        return replace(value, operations=tuple(operations))

    return replace(program, blocks=(block(program.blocks[0], "entry"),))


def seed_origins(program):
    """Assign deterministic original-SSA IDs once, preserving existing histories."""
    ssa.verify_program(program)

    if "provenance" in program.metadata:
        _metadata(program)

        return program

    locations = operation_locations(program)

    if any(operation.origins for _location, operation in locations):
        raise ValueError("Cannot seed operations with origins but no source catalog.")

    namespace = _fingerprint(program)
    sources = {
        f"{namespace}:{location}": {
            "location": location,
            "opcode": operation.opcode,
            "results": tuple(result.name for result in operation.results),
        }
        for location, operation in locations
    }
    seeded = _map_locations(
        program,
        lambda location, operation: replace(
            operation, origins=(f"{namespace}:{location}",)
        ),
    )

    return replace(
        seeded,
        metadata=dict(program.metadata)
        | {
            "provenance": {
                "schema": 1,
                "namespace": namespace,
                "sources": sources,
                "passes": (),
            }
        },
    )


class ProvenancePass:
    """Record explicit relations for a single immutable Program-to-Program pass."""

    def __init__(self, before, name):
        self.before = before
        self.name = str(name)
        self.data = _metadata(before)
        self.inputs = operation_locations(before)
        self._input_locations = {}

        for location, operation in self.inputs:
            self._input_locations.setdefault(id(operation), []).append(location)

        self._relations = []
        self._result_mappings = []

    def _source_locations(self, sources):
        locations = []

        for source in sources:
            matches = self._input_locations.get(id(source), ())

            if len(matches) != 1:
                raise ValueError(
                    "Each declared source must be one unique input operation."
                )

            locations.append(matches[0])

        if len(set(locations)) != len(locations):
            raise ValueError("A provenance relation cannot repeat a source operation.")
        return tuple(locations)

    def derive(self, targets, sources, *, relation="replace", recursive=False):
        """Attach source candidates and record an explicit replacement relation.

        ``recursive=True`` declares every generated nested operation as part of
        the same transformation. Partially unknown inputs produce unknown origins
        rather than presenting a known subset as a complete source set.
        """
        sources, targets = tuple(sources), tuple(targets)

        if not sources or not targets:
            raise ValueError(
                "Use delete for deletions; derivations need sources and targets."
            )

        if relation not in {"replace", "split", "merge"}:
            raise ValueError("A derivation relation must be replace, split or merge.")

        locations = self._source_locations(sources)

        if any(
            kind == "delete" and set(locations).intersection(previous_sources)
            for kind, previous_sources, _targets, _origins in self._relations
        ):
            raise ValueError("A deleted input cannot also have a derived successor.")

        origins = (
            tuple(sorted({origin for source in sources for origin in source.origins}))
            if all(source.origins for source in sources)
            else ()
        )
        generated = []

        def derive_operation(operation):
            if recursive:
                operation = replace(
                    operation,
                    regions=tuple(
                        replace(
                            region,
                            operations=tuple(map(derive_operation, region.operations)),
                        )
                        for region in operation.regions
                    ),
                )

            operation = replace(operation, origins=origins)
            generated.append(operation)

            return operation

        result = tuple(map(derive_operation, targets))

        if relation == "split" and len(generated) < 2:
            raise ValueError("A split must declare at least two generated operations.")

        if relation == "merge" and len(sources) < 2:
            raise ValueError("A merge must declare at least two source operations.")

        self._relations.append((relation, locations, tuple(generated), origins))

        return result

    def map_result(
        self,
        source,
        target,
        *,
        source_result=None,
        target_result=None,
        projection="identity",
    ):
        """Declare a result equality to check, separately from operation origins.

        This is a pass author's semantic contract, not proof of equivalence.
        ``lane`` compares a numeric reference tile at the candidate's scalar
        lane. Neither names nor split/merge origins imply such a contract.
        """
        self._source_locations((source,))

        def select(operation, name):
            if name is None and len(operation.results) == 1:
                return operation.results[0]

            matches = [value for value in operation.results if value.name == name]

            if len(matches) != 1:
                raise ValueError("A result mapping must name one actual SSA result.")
            return matches[0]

        left, right = select(source, source_result), select(target, target_result)

        if projection == "identity":
            compatible = left.type == right.type
        elif projection in {
            "lane",
            "matmul_lhs",
            "matmul_rhs",
            "matmul_term",
            "matmul_prefix",
        }:
            dtype = left.type.dtype

            if projection.startswith("matmul_"):
                if (
                    source.opcode not in {"linalg.matmul", "linalg.dot"}
                    or len(source.operands) != 2
                ):
                    raise ValueError(
                        "Matmul checkpoints require a two-operand dot/matmul source."
                    )

                if projection in {"matmul_lhs", "matmul_rhs"}:
                    values = {value.name: value.type for value in self.before.inputs}
                    values.update(
                        (value.name, value.type)
                        for _, operation in self.inputs
                        for value in operation.results
                    )
                    values.update(
                        (value.name, value.type)
                        for block in self.before.blocks
                        for value in block.args
                    )
                    values.update(
                        (value.name, value.type)
                        for _, operation in self.inputs
                        for region in operation.regions
                        for value in region.args
                    )
                    operand = 0 if projection == "matmul_lhs" else 1
                    dtype = values[source.operands[operand]].dtype

            compatible = (
                left.type.kind == "tensor"
                and right.type.kind == "scalar"
                and dtype == right.type.dtype
                and not right.type.shape
            )
        else:
            raise ValueError(
                "Result projection must be identity, lane, or a supported matmul checkpoint."
            )

        if not compatible or left.type.kind in {"pointer", "tuple"}:
            raise ValueError("Result mapping types are incompatible.")

        for (
            _previous_source,
            previous_target,
            _previous_left,
            previous_right,
            _,
        ) in self._result_mappings:
            if target is previous_target and right.name == previous_right:
                raise ValueError(
                    "A result cannot have ambiguous or duplicate mappings."
                )

        self._result_mappings.append(
            (source, target, left.name, right.name, projection)
        )

    def delete(self, *sources):
        """Declare deletion, reporting original sources only when all are known."""
        if not sources:
            raise ValueError("A deletion must name at least one input operation.")

        locations = self._source_locations(sources)

        if any(
            set(locations).intersection(previous_sources)
            for _kind, previous_sources, _targets, _origins in self._relations
        ):
            raise ValueError("A deletion conflicts with an existing source relation.")

        origins = (
            tuple(sorted({origin for source in sources for origin in source.origins}))
            if all(source.origins for source in sources)
            else ()
        )
        self._relations.append(("delete", locations, (), origins))

    def finish(self, after):
        """Append a pass record; unrecorded replacements and removals stay unknown."""
        outputs = operation_locations(after)
        output_locations = {}

        for location, operation in outputs:
            output_locations.setdefault(id(operation), []).append(location)

        records, handled_inputs, handled_outputs, declared_origins = (
            [],
            set(),
            set(),
            {},
        )

        for relation, source_locations, targets, origins in self._relations:
            target_locations = []

            for target in targets:
                matches = output_locations.get(id(target), ())

                if len(matches) != 1:
                    raise ValueError(
                        "Each declared target must occur once in the output program."
                    )

                location = matches[0]

                if location in handled_outputs:
                    raise ValueError(
                        "An output operation has more than one provenance relation."
                    )

                target_locations.append(location)
                declared_origins[location] = origins
                handled_outputs.add(location)

            if relation == "delete" and any(
                source is target
                for location, source in self.inputs
                if location in source_locations
                for _target_location, target in outputs
            ):
                raise ValueError(
                    "A deleted operation is still present in the output program."
                )

            handled_inputs.update(source_locations)
            records.append(
                {
                    "relation": relation,
                    "inputs": source_locations,
                    "outputs": tuple(target_locations),
                    "origins": origins,
                }
            )

        for location, operation in outputs:
            if location in handled_outputs:
                continue

            matches = self._input_locations.get(id(operation), ())

            if len(matches) == 1 and len(output_locations[id(operation)]) == 1:
                source_location = matches[0]
                origins = operation.origins
                relation, source_locations = "preserve", (source_location,)
                handled_inputs.add(source_location)
            else:
                origins, relation, source_locations = (), "unmapped", ()

            declared_origins[location] = origins
            records.append(
                {
                    "relation": relation,
                    "inputs": source_locations,
                    "outputs": (location,),
                    "origins": origins,
                }
            )

        for location, operation in self.inputs:
            if location not in handled_inputs:
                records.append(
                    {
                        "relation": "unmapped_removal",
                        "inputs": (location,),
                        "outputs": (),
                        "origins": operation.origins,
                    }
                )

        normalized = _map_locations(
            after,
            lambda location, operation: replace(
                operation, origins=declared_origins[location]
            ),
        )
        entry = {
            "name": self.name,
            "input_fingerprint": _fingerprint(self.before),
            "output_fingerprint": _fingerprint(after),
            "relations": tuple(records),
        }
        mappings = []

        for source, target, left, right, projection in self._result_mappings:
            (source_location,) = self._source_locations((source,))
            target_locations = output_locations.get(id(target), ())

            if len(target_locations) != 1:
                raise ValueError(
                    "A mapped result's producer must occur once in the output."
                )

            target_location = target_locations[0]

            if not any(
                source_location in record["inputs"]
                and target_location in record["outputs"]
                and record["relation"] in {"preserve", "replace", "split", "merge"}
                for record in records
            ):
                raise ValueError(
                    "Result mapping requires an explicit operation relation."
                )

            mappings.append(
                {
                    "source_location": source_location,
                    "target_location": target_location,
                    "source_result": left,
                    "target_result": right,
                    "projection": projection,
                }
            )

        if mappings:
            entry["value_mappings"] = tuple(mappings)

        return replace(
            normalized,
            metadata=dict(after.metadata)
            | {
                "provenance": dict(self.data)
                | {"passes": (*self.data["passes"], entry)}
            },
        )


def record_pass(before, after, name):
    """Wrap a pass, accepting its explicit record or recording retained objects.

    A transforming pass should use ProvenancePass itself. This wrapper does not
    infer replacement mappings even when names, structure or copied IDs match.
    """
    tracker = ProvenancePass(before, name)
    data = after.metadata.get("provenance")

    if data is not None and data.get("schema") == 1:
        history = data.get("passes", ())
        previous = tracker.data["passes"]

        if (
            data.get("namespace") == tracker.data["namespace"]
            and data.get("sources") == tracker.data["sources"]
            and len(history) == len(previous) + 1
            and tuple(history[:-1]) == tuple(previous)
            and history[-1]["name"] == str(name)
            and history[-1]["input_fingerprint"] == _fingerprint(before)
            and history[-1]["output_fingerprint"] == _fingerprint(after)
        ):
            _metadata(after)

            return after
    return tracker.finish(after)


@dataclass(frozen=True)
class SourceCandidate:
    """A declared original-SSA source; neither value equivalence nor fault proof."""

    origin_id: str
    location: str
    opcode: str
    result_names: tuple[str, ...]


def source_candidates(program, locations=None):
    """Resolve origins for given locations or the last pass's changed relations.

    Without locations this includes explicit split, merge, replacement and
    deletion sources, and excludes retained operations and unknown mappings.
    """
    if "provenance" not in program.metadata:
        return ()

    data = _metadata(program)

    if locations is not None:
        requested = set(locations)
        origins = {
            origin
            for location, operation in operation_locations(program)
            if location in requested
            for origin in operation.origins
        }
    else:
        history = data["passes"]
        relations = history[-1]["relations"] if history else ()
        origins = {
            origin
            for record in relations
            if record["relation"] in {"replace", "split", "merge", "delete"}
            for origin in record["origins"]
        }
    return tuple(
        SourceCandidate(
            origin,
            data["sources"][origin]["location"],
            data["sources"][origin]["opcode"],
            tuple(data["sources"][origin]["results"]),
        )
        for origin in sorted(origins)
    )
