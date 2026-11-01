"""Map bounded physical launches onto one unchanged logical task domain."""

import re
from dataclasses import dataclass


def allocate_private_symbol(base, *fragments):
    """Keep a private argument or local distinct from emitted identifiers."""
    identifiers = set(re.findall(r"\b[A-Za-z_]\w*\b", "\n".join(fragments)))
    name = base
    suffix = 0

    while name in identifiers:
        suffix += 1
        name = f"{base}_{suffix}"

    return name


@dataclass(frozen=True)
class TaskLaunchSymbols:
    offset: str
    total: str
    logical: str
    segment: str
    physical: str
    error: str
    grid: str
    kind: str

    @property
    def parameters(self):
        return (f"int64_t {self.offset}", f"int64_t {self.total}")

    def kernel_body(self, body, *, task_pattern, physical_index, max_tasks):
        """Guard the remaining range before adding its logical offset."""
        body = re.sub(task_pattern, self.logical, body)

        return f"""    const int64_t {self.physical} = (int64_t){physical_index};
    if ({self.offset} < 0 || {self.total} <= {self.offset}) {{ return; }}
    if ({self.physical} >= {max_tasks}LL || {self.physical} >= {self.total} - {self.offset}) {{ return; }}
    const int64_t {self.logical} = {self.offset} + {self.physical};
    if ({self.logical} >= {self.total}) {{ return; }}
{body}"""


def allocate_task_launch_symbols(*source_fragments):
    """Keep internal launch arguments distinct from all emitted identifiers."""
    identifiers = set(re.findall(r"\b[A-Za-z_]\w*\b", "\n".join(source_fragments)))
    names = []

    for role in (
        "offset",
        "total",
        "logical",
        "segment",
        "physical",
        "error",
        "grid",
        "kind",
    ):
        base = f"nt_launch_{role}"
        name = base
        suffix = 0

        while name in identifiers:
            suffix += 1
            name = f"{base}_{suffix}"

        identifiers.add(name)
        names.append(name)

    return TaskLaunchSymbols(*names)
