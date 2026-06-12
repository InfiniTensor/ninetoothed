"""Debug tensor naming and stride hint matching."""
import ninetoothed, pathlib, re
from ninetoothed import Symbol, Tensor
from ninetoothed.generation import CodeGenerator, TilingHint
import ninetoothed.naming as naming

BS = Symbol('BS', meta=True, lower_bound=256, upper_bound=256)

def arrangement(x, output):
    return x.tile((BS,)), output.tile((BS,))

def application(x, output):
    output = x  # noqa

# Fixed names for reproducibility
t1 = Tensor(1, name='input_a')
t2 = Tensor(1, name='input_b')
tensors = (t1, t2)

print('t1 source name:', t1.source.name)
print('t1 bare name:', naming.remove_prefixes(t1.source.name))
print('t2 source name:', t2.source.name)
print('t2 bare name:', naming.remove_prefixes(t2.source.name))

# Use ninetoothed.make to set annotations
k = ninetoothed.make(arrangement, application, tensors, kernel_name='dbg_base')
src_base = pathlib.Path(k._source).read_text()

# Generate hinted version with correct tensor names
bare1 = naming.remove_prefixes(t1.source.name)
bare2 = naming.remove_prefixes(t2.source.name)
hint = TilingHint(
    has_divisible_tiles=True,
    contiguous_dims={(bare1, 0), (bare2, 0)},
    known_strides={(bare1, 0): 1, (bare2, 0): 1},
    exact_innermost_sizes=True,
)
gen = CodeGenerator(tiling_hint=hint)
f = gen(application, 'torch', 'dbg_hint', 4, 3, 1, False)
src = open(f).read()

# Show body only
for label, text in [('BASELINE', src_base), ('HINTED', src)]:
    lines = text.splitlines()
    body_start = next(i+1 for i, l in enumerate(lines) if l.strip().startswith('def '))
    body = '\n'.join(lines[body_start:])
    print(f'\n=== {label} ===')
    print(body)
    stride_rx = r'_stride_\d+'
    print(f'  stride refs: {len(re.findall(stride_rx, body))}')
    and_count = body.count(' & ')
    print(f'  mask complexity: {and_count}')
