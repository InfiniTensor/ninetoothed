# Nine-Toothed

A domain-specific language (DSL) based on Triton but providing higher-level abstractions.

**Other language versions: [English](README.md), [Simplified Chinese](docs/README.zh.md).**

## Installation

Since `ninetoothed` is not yet available on PyPI, we need to manually `git clone` this repository and then `pip install`.

```shell
git clone https://github.com/InfiniTensor/ninetoothed.git
pip install ./ninetoothed
```

After successfully running the above two commands, `ninetoothed` will be installed. Note that we haven't added any dependency management yet, so if you want to run the compiled kernel functions, you should at least install Triton.

```shell
pip install triton
```

Other packages can be installed as needed, such as `torch`, `matplotlib`, `pandas`, etc.

## Usage

Currently, we can use the `Tensor` and `Symbol` classes in the `ninetoothed` package to perform meta-operations like `tile` and `expand` to easily construct kernel functions. Below, we will use these features to create vector addition and matrix multiplication kernel functions.

### Vector Addition

```python
BLOCK_SIZE = Symbol("BLOCK_SIZE", meta=True)

@ninetoothed.jit
def add_kernel(
    x: Tensor(1).tile((BLOCK_SIZE,)),
    y: Tensor(1).tile((BLOCK_SIZE,)),
    z: Tensor(1).tile((BLOCK_SIZE,)),
):
    z = x + y
```

In this code, we first define `BLOCK_SIZE`, which is a `Symbol`. You can think of `"BLOCK_SIZE"` as its name. We see that `meta` is set to `True`, indicating to the compiler that it is a meta-parameter and its value can be determined by the compiler. The `Tensor(1)` constructs a one-dimensional tensor (vector), and `Tensor(1).tile((BLOCK_SIZE,))` means we want to create a vector and divide it into blocks of size `BLOCK_SIZE`. Suppose the size of this vector is `8192` and `BLOCK_SIZE` is `1024`, then the vector will be divided into `8` blocks, each of size `1024`.

By using type annotations, we tell the compiler that we will have three tensor parameters, which will be divided into blocks, and `x`, `y`, and `z` are these blocks. It's important to understand that `x`, `y`, and `z` are the blocks, not the tensors themselves. In the function body, `x`, `y`, and `z` are also the blocks. The rest is straightforward (only one line `z = x + y` left, haha), we add each block of `x` and `y` and store it in `z`. Since each block of the parameter tensors undergoes this operation, the addition is completed for the whole tensors as well.

### Matrix Multiplication

```python
BLOCK_SIZE_M = Symbol("BLOCK_SIZE_M", meta=True)
BLOCK_SIZE_N = Symbol("BLOCK_SIZE_N", meta=True)
BLOCK_SIZE_K = Symbol("BLOCK_SIZE_K", meta=True)

a_tiled = Tensor(2).tile((BLOCK_SIZE_M, BLOCK_SIZE_K)).tile((1, -1))
b_tiled = Tensor(2).tile((BLOCK_SIZE_K, BLOCK_SIZE_N)).tile((-1, 1))
c_tiled = Tensor(2).tile((BLOCK_SIZE_M, BLOCK_SIZE_N))

a_tiled = a_tiled.expand((-1, c_tiled.shape[1]))
b_tiled = b_tiled.expand((c_tiled.shape[0], -1))

@ninetoothed.jit
def matmul_kernel(a: a_tiled, b: b_tiled, c: c_tiled):
    accumulator = ninetoothed.language.zeros(
        c.shape, dtype=ninetoothed.language.float32
    )
    for k in range(a.shape[1]):
        accumulator = ninetoothed.language.dot(a[0, k], b[k, 0], accumulator)
    c = accumulator.to(ninetoothed.language.float16)
```

For matrix multiplication, we also have three tensor parameters, but the tiling method is more complex than vector addition. We denote the three matrices as $A$, $B$, and $C$, where $A$ and $B$ are inputs, and $C$ is the output. Tiling $C$ is simple; we just need to divide it into blocks of size `(BLOCK_SIZE_M, BLOCK_SIZE_N)` by rows and columns. Once each block computes its result, the entire $C$ is computed. However, how should we tile $A$ and $B$? The answer is to introduce another meta-parameter `BLOCK_SIZE_K`. This way, we can divide $A$ into blocks of size `(BLOCK_SIZE_M, BLOCK_SIZE_K)` and $B$ into blocks of size `(BLOCK_SIZE_K, BLOCK_SIZE_N)`. However, for matrix multiplication, $A$ and $B$ do not correspond block by block; each row of $A$ needs to correspond to each column of $B$. Therefore, we need to further `tile` $A$ and $B$ by rows and columns, respectively. Up to this point, we have a set of row blocks of $A$ and column blocks of $B$. However, each row block of $A$ must correspond to every column block of $B$. This is where `expand` comes in. We `expand` the row blocks of $A$ along the columns to the number of columns of $C$ and the column blocks of $B$ along the rows to the number of rows of $C$. This way, we successfully tile $A$, $B$, and $C$.

With tiling done, the rest is simple. In the function body, we define an `accumulator` to accumulate intermediate results. We then iterate through the corresponding row blocks of $A$ and column blocks of B, multiplying them and accumulating the results in `accumulator`. Finally, we place the `accumulator` in the corresponding block of $C$. Since each block of the parameter tensors undergoes this operation, the multiplication is completed for the whole tensors as well.
