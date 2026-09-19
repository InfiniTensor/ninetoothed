"""Check declared numeric results and explain their observed value dependencies."""

import hashlib
from dataclasses import dataclass

import numpy as np

from ninetoothed.ir import ssa
from ninetoothed.ir.provenance import operation_locations

from .access import WrittenIntervals, merge_ranges


@dataclass(frozen=True)
class MemoryDependency:
    """An observed read of bytes last written by one same-program event."""

    reader_index: int
    writer_index: int
    storage: str
    byte_ranges: tuple


@dataclass(frozen=True)
class ResultMismatch:
    """A failed declared equality, not a proof of unique causal blame."""

    reference_index: int
    candidate_index: int
    reference_result: str
    candidate_result: str
    projection: str = "identity"


@dataclass(frozen=True)
class DependencyEvent:
    """One executed producer on a backward value slice."""

    trace_index: int
    location: str
    opcode: str
    program_id: tuple
    iteration: tuple
    lane: tuple | None


@dataclass(frozen=True)
class DependencySlice:
    """Observed SSA value edges with explicit memory/control-flow boundaries."""

    events: tuple[DependencyEvent, ...]
    boundaries: tuple[str, ...]
    scope: str = "Observed value dependencies; not a minimal slice or unique cause."
    memory_dependencies: tuple[MemoryDependency, ...] = ()


def _memory_edges(trace):
    history, edges = {}, {}
    links, boundaries = [set() for _ in trace], [set() for _ in trace]
    complete = True
    known = True

    for index, event in enumerate(trace):
        complete = complete and event.sequence == index

        for access in event.memory or ():
            if access.kind == "unknown":
                known = False

                continue

            intervals = history.setdefault(access.storage, WrittenIntervals())

            if access.kind == "write":
                if access.overlapping_lanes:
                    boundaries[index].add(
                        f"{event.location}: overlapping write lanes have no unique parallel order"
                    )

                for start, end in access.byte_ranges:
                    intervals.write(start, end, index)

                continue

            if not complete or not known:
                reason = (
                    "filtered or incomplete trace"
                    if not complete
                    else "uninstrumented handler effects"
                )
                boundaries[index].add(
                    f"{event.location}: memory history unavailable after {reason}"
                )

                continue

            for start, end in access.byte_ranges:
                for lower, upper, writer in intervals.readers(start, end):
                    if trace[writer].program_id != event.program_id:
                        boundaries[index].add(
                            f"{event.location}: cross-program memory order is only sequential CPU order"
                        )
                    elif writer != index:
                        links[index].add(writer)
                        edges.setdefault((index, writer, access.storage), []).append(
                            (lower, upper)
                        )

    observed = tuple(
        MemoryDependency(*key, merge_ranges(ranges))
        for key, ranges in sorted(edges.items())
    )

    return links, boundaries, observed


def _contexts(program):
    contexts = {}

    def visit(block, path, parents):
        for index, operation in enumerate(block.operations):
            location = f"{path}:{index}:{operation.opcode}"
            contexts[location] = parents

            for region_index, region in enumerate(operation.regions):
                header = (
                    operation.opcode,
                    operation.operands,
                    operation.attrs,
                    region_index,
                    region.args,
                )
                visit(region, f"{location}/region{region_index}", (*parents, header))

    visit(program.blocks[0], "entry", ())

    return contexts


def _trace_by_location(trace):
    locations = {}

    for index, event in enumerate(trace):
        locations.setdefault(event.location, []).append((index, event))
    return locations


def _matmul_checkpoints(operation, left, right, result_name, projection):
    """Derive scalar K checkpoints from original input tiles, never candidates.

    Each active candidate lane must execute exactly the original 0..K-1
    sequence. Accumulation casts at each step to the declared output dtype.
    This is a scalar decomposition contract, not arbitrary reassociation.
    """
    if (
        operation.opcode not in {"linalg.dot", "linalg.matmul"}
        or len(operation.operands) != 2
    ):
        raise ValueError("Source is not a two-operand matrix product.")

    catalog, arrays = {}, {}

    for index, event in left:
        key = event.program_id, event.iteration

        if key in catalog or event.lane is not None:
            raise ValueError("Reference matrix events are ambiguous.")

        tiles = []

        for name in operation.operands:
            snapshot = (event.inputs or {}).get(name)

            if not isinstance(snapshot, dict) or "value" not in snapshot:
                raise ValueError("Reference input tiles were not captured.")

            array = np.asarray(snapshot["value"], dtype=snapshot["dtype"])

            if array.ndim != 2 or list(array.shape) != snapshot["shape"]:
                raise ValueError("Reference inputs must be complete rank-two tiles.")

            tiles.append(array)

        if tiles[0].shape[1] != tiles[1].shape[0]:
            raise ValueError("Reference reduction dimensions differ.")

        result = event.results.get(result_name)

        if not isinstance(result, dict) or "value" not in result:
            raise ValueError("Reference matrix result was not captured.")

        catalog[key] = (index, event)
        arrays[index] = (*tiles, np.dtype(result["dtype"]))

    groups, pairs = {}, []

    for index, event in right:
        if not event.iteration or event.lane is None or len(event.lane) != 2:
            raise ValueError("Checkpoint needs one K iteration and a matrix lane.")

        key = event.program_id, event.iteration[:-1]

        if key not in catalog:
            raise ValueError("Checkpoint has no corresponding reference tile.")

        reference = catalog[key]
        a, b, _dtype = arrays[reference[0]]
        row, column = event.lane

        if not (0 <= row < a.shape[0] and 0 <= column < b.shape[1]):
            raise ValueError("Checkpoint lane is outside its reference tile.")

        group = reference[0], event.lane
        groups.setdefault(group, []).append(event.iteration[-1])
        pairs.append((reference, (index, event)))

    active = {key for key, (index, _) in catalog.items() if arrays[index][0].shape[1]}

    if active != {(event.program_id, event.iteration[:-1]) for _, event in right}:
        raise ValueError("Checkpoint coverage differs from nonempty reference tiles.")

    for (index, _lane), iterations in groups.items():
        if iterations != list(range(arrays[index][0].shape[1])):
            raise ValueError(
                "Checkpoint iterations must cover K exactly once in order."
            )

    if not pairs:
        return None

    expected, accumulators = [], {}

    for reference, candidate in pairs:
        index, _event = reference
        _candidate_index, event = candidate
        a, b, dtype = arrays[index]
        row, column = event.lane
        k = event.iteration[-1]

        if projection == "matmul_lhs":
            value = np.asarray(a[row, k])
        elif projection == "matmul_rhs":
            value = np.asarray(b[k, column])
        else:
            value = np.asarray(np.multiply(a[row, k], b[k, column]), dtype=dtype)

            if projection == "matmul_prefix":
                key = index, event.lane
                previous = accumulators.get(key, np.zeros((), dtype=dtype))
                value = np.asarray(previous + value, dtype=dtype)
                accumulators[key] = value

        expected.append(
            (
                reference,
                candidate,
                {"dtype": str(value.dtype), "shape": [], "value": value.item()},
            )
        )
    return expected


def compare_mapped_results(
    reference, candidate, first, second, same_snapshot, rtol, atol
):
    """Compare explicitly declared results only at matching execution contexts.

    All occurrences of a mapping must align before any of its mismatches is
    reported. A lane projection checks executed scalar lanes against a numeric
    reference tile; it does not claim coverage of inactive padded lanes.
    """
    history = candidate.metadata.get("provenance", {}).get("passes", ())

    if not history or not history[-1].get("value_mappings"):
        return None, ()

    latest = history[-1]

    for key, program in (
        ("input_fingerprint", reference),
        ("output_fingerprint", candidate),
    ):
        if latest.get(key) != hashlib.sha256(ssa.render(program).encode()).hexdigest():
            return None, (
                "Result mapping does not match the compared SSA fingerprints.",
            )

    before, after = (
        dict(operation_locations(reference)),
        dict(operation_locations(candidate)),
    )
    left_contexts, right_contexts = _contexts(reference), _contexts(candidate)
    left_events, right_events = (
        _trace_by_location(first.trace),
        _trace_by_location(second.trace),
    )
    relations = {}

    for relation in latest["relations"]:
        if relation["relation"] in {"preserve", "replace", "split", "merge"}:
            for location in relation["outputs"]:
                relations.setdefault(location, set()).update(relation["inputs"])

    mismatches, issues, used_targets = [], [], set()

    for mapping in latest["value_mappings"]:
        source, target = mapping["source_location"], mapping["target_location"]
        left_name, right_name = mapping["source_result"], mapping["target_result"]
        right_key = (target, right_name)

        if right_key in used_targets:
            return None, ("Duplicate result mapping is ambiguous.",)

        used_targets.add(right_key)

        if source not in before or target not in after:
            issues.append(f"Missing result producer: {source} -> {target}.")
            continue

        projection = mapping["projection"]
        contexts = right_contexts[target]

        if projection.startswith("matmul_"):
            if (
                projection
                not in {"matmul_lhs", "matmul_rhs", "matmul_term", "matmul_prefix"}
                or not contexts
                or contexts[-1][0] != "scf.for"
                or contexts[-1][2].get("decomposition") != "matmul"
                or contexts[-1][3] != 0
                or len(contexts[-1][4]) != 2
            ):
                issues.append(
                    f"Unsupported matmul checkpoint context: {source} -> {target}."
                )
                continue

            contexts = contexts[:-1]

        if left_contexts[source] != contexts:
            issues.append(f"Changed enclosing control flow: {source} -> {target}.")
            continue

        if source not in relations.get(target, ()):
            issues.append(f"Missing operation relation: {source} -> {target}.")
            continue

        left, right = left_events.get(source, ()), right_events.get(target, ())

        if not left and not right:
            continue

        pairs = []

        if projection == "identity":

            def key(event):
                return event.program_id, event.iteration, event.lane

            left_keys, right_keys = (
                [key(event) for _, event in left],
                [key(event) for _, event in right],
            )

            if left_keys == right_keys and len(set(left_keys)) == len(left_keys):
                pairs = [(lhs, rhs, None) for lhs, rhs in zip(left, right)]
        elif projection == "lane" and all(event.lane is None for _, event in left):
            catalog = {
                (event.program_id, event.iteration): (index, event)
                for index, event in left
            }
            right_keys = [
                (event.program_id, event.iteration, event.lane) for _, event in right
            ]

            if (
                len(catalog) == len(left)
                and len(set(right_keys)) == len(right_keys)
                and set(catalog)
                == {(event.program_id, event.iteration) for _, event in right}
                and all(event.lane is not None for _, event in right)
            ):
                pairs = [
                    (catalog[event.program_id, event.iteration], (index, event), None)
                    for index, event in right
                ]
        elif projection.startswith("matmul_"):
            try:
                pairs = _matmul_checkpoints(
                    before[source], left, right, left_name, projection
                )
            except (ValueError, TypeError, KeyError) as exc:
                issues.append(f"Invalid matmul checkpoint: {source} -> {target}: {exc}")
                continue

            if pairs is None:
                continue

        if not pairs:
            issues.append(
                f"Result events unavailable or unaligned: {source} -> {target}."
            )
            continue

        local_mismatches = []

        for (left_index, left_event), (right_index, right_event), expected in pairs:
            lhs, rhs = (
                left_event.results.get(left_name) if expected is None else expected,
                right_event.results.get(right_name),
            )

            if (
                not isinstance(lhs, dict)
                or not isinstance(rhs, dict)
                or "value" not in lhs
                or "value" not in rhs
            ):
                issues.append(
                    f"Result is not a numeric snapshot: {source} -> {target}."
                )
                break

            if projection == "lane":
                array = np.asarray(lhs["value"], dtype=lhs["dtype"])
                lane = right_event.lane

                if (
                    rhs["shape"] != []
                    or len(lane) != array.ndim
                    or any(i < 0 or i >= size for i, size in zip(lane, array.shape))
                ):
                    issues.append(
                        f"Invalid scalar lane projection: {source} -> {target}."
                    )
                    break

                lhs = {"dtype": lhs["dtype"], "shape": [], "value": array[lane].item()}

            if not same_snapshot(lhs, rhs, rtol, atol):
                local_mismatches.append(
                    ResultMismatch(
                        left_index, right_index, left_name, right_name, projection
                    )
                )
        else:
            mismatches.extend(local_mismatches)
    return min(mismatches, key=lambda item: item.candidate_index, default=None), tuple(
        issues
    )


def backward_slice(program, trace, observation, *, memory=True):
    """Follow value, selected-yield and loop-carried execution dependencies.

    Program ID, scalar lane and iteration scope keep separate executions apart.
    Checked byte accesses connect same-program reads to their last writes.
    Gaps, unknown effects and cross-program ordering remain explicit boundaries.
    ``memory=False`` retains the original diagnostics for saved legacy bundles.
    """
    indices = [
        index
        for index, event in enumerate(trace)
        if (
            event.location == observation.location
            and event.program_id == observation.program_id
            and event.iteration == observation.iteration
            and event.lane == observation.lane
        )
    ]

    if len(indices) != 1:
        return DependencySlice((), ("Observation has no unique execution event.",))

    # Later events cannot supply a value read by this observation.
    trace = trace[: indices[0] + 1]
    operations = dict(operation_locations(program))
    arguments, parents = {}, {}

    def describe(block, path, enclosing=(), loop_depth=0):
        for offset, op in enumerate(block.operations):
            location = f"{path}:{offset}:{op.opcode}"
            parents[location] = enclosing

            for region_index, region in enumerate(op.regions):
                depth = loop_depth + (op.opcode == "scf.for")

                for index, value in enumerate(region.args):
                    arguments[value.name] = (location, op, index, depth)

                describe(
                    region,
                    f"{location}/region{region_index}",
                    (*enclosing, (location, op, depth)),
                    depth,
                )

    describe(program.blocks[0], "entry")
    memory_links, memory_boundaries, memory_dependencies = (
        _memory_edges(trace)
        if memory and any(event.memory for event in trace)
        else (None, None, ())
    )
    definitions, yields, edges, boundaries = {}, {}, [], []
    roots = {value.name for value in program.inputs} | set(
        program.metadata.get("symbols", ())
    )

    for index, event in enumerate(trace):
        operation = operations[event.location]
        links, stops = (
            (set(), set())
            if memory_links is None
            else (memory_links[index], memory_boundaries[index])
        )

        def resolve(name, seen=()):
            if name in seen:
                stops.add(f"{event.location}: cyclic value dependency {name}")

                return

            producer = definitions.get((event.program_id, event.lane, name))

            if producer is not None:
                origin = trace[producer]

                if event.iteration[: len(origin.iteration)] == origin.iteration:
                    links.add(producer)

                    return

            if name in arguments:
                location, parent, position, depth = arguments[name]

                if parent.opcode == "scf.for" and len(event.iteration) >= depth:
                    if position == 0:
                        # The induction value is determined by the loop schedule.
                        for operand in parent.operands[:3]:
                            resolve(operand, (*seen, name))
                    else:
                        prior = yields.get(
                            (
                                event.program_id,
                                event.lane,
                                location,
                                event.iteration[: depth - 1],
                            )
                        )

                        if prior is not None:
                            links.add(prior)
                        else:
                            resolve(parent.operands[position + 2], (*seen, name))
                    return

            if name not in roots:
                stops.add(
                    f"{event.location}: unresolved region argument or value {name}"
                )

        names = set(operation.operands) | set(operation.attrs.get("indices", ()))

        for name in names:
            resolve(name)

        for _location, parent, _depth in parents[event.location]:
            for name in parent.operands[: 3 if parent.opcode == "scf.for" else 1]:
                resolve(name)

        if operation.regions:
            yielded = yields.get(
                (event.program_id, event.lane, event.location, event.iteration)
            )

            if yielded is not None:
                links.add(yielded)
            elif operation.opcode not in {"scf.for", "scf.if"}:
                stops.add(f"{event.location}: unsupported region-result dependency")

        if (not memory or event.memory is None) and operation.opcode in {
            "mem.load",
            "mem.store",
            "tensor.extract",
        }:
            stops.add(f"{event.location}: memory/alias history not reconstructed")

        edges.append(links)
        boundaries.append(stops)

        for result in operation.results:
            definitions[event.program_id, event.lane, result.name] = index

        if operation.opcode == "scf.yield" and parents[event.location]:
            location, parent, depth = parents[event.location][-1]
            outer = (
                event.iteration[: depth - 1]
                if parent.opcode == "scf.for"
                else event.iteration
            )
            yields[event.program_id, event.lane, location, outer] = index

    visited, pending = set(), list(indices)

    while pending:
        index = pending.pop()

        if index not in visited:
            visited.add(index)
            pending.extend(edges[index])

    events = tuple(
        DependencyEvent(
            index,
            trace[index].location,
            trace[index].opcode,
            trace[index].program_id,
            trace[index].iteration,
            trace[index].lane,
        )
        for index in sorted(visited)
    )

    return DependencySlice(
        events,
        tuple(sorted({message for index in visited for message in boundaries[index]})),
        memory_dependencies=tuple(
            edge for edge in memory_dependencies if edge.reader_index in visited
        ),
    )
