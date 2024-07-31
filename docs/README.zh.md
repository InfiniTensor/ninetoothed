# 九齿

一种基于 Triton 但提供更高层抽象的领域特定语言（DSL）。

**其他语言版本: [English](../README.md)、[简体中文](README.zh.md)。**

## 安装

由于现在 `ninetoothed` 还没有进入到 PyPI，我们需要手动 `git clone` 本仓库，再 `pip install`。

```shell
git clone https://github.com/InfiniTensor/ninetoothed.git
pip install ./ninetoothed
```

成功运行完以上两个命令之后，`ninetoothed` 就被安装好了。需要注意的是，我们目前还没有加入任何的依赖管理，所以如果想要运行编译好的核函数，那么最起码应该再安装 Triton。

```shell
pip install triton
```

其余包可以根据需要自行安装，如 `torch`、`matplotlib`、`pandas` 等。

## 使用

目前，我们可以通过 `ninetoothed` 包当中的 `Tensor` 和 `Symbol` 类，进行 `tile` 和 `expand` 等元操作，从而简单地构建核函数。下面，我们将使用这些内容构建出向量加法和矩阵乘法核函数。

### 向量加法

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

在这段代码当中，我们首先定义了 `BLOCK_SIZE`，它是一个 `Symbol`，我们可以把 `"BLOCK_SIZE"` 理解成它的名字。我们可以看到 `meta` 被设成了 `True`，这是在告诉编译器，它是一个元参数，可以由编译器决定它的取值。之后出现的 `Tensor(1)` 则是在构造一个一维的张量（向量），`Tensor(1).tile((BLOCK_SIZE,))` 的意思就是说，我们想要构造一个向量，并且把它分成大小为 `BLOCK_SIZE` 的块。假设这个向量的大小为 `8192`，而 `BLOCK_SIZE` 是 `1024`，那么这个向量就会被分成 `8` 块，每一块的大小都是 `1024`。

我们通过类型标注的方式，告诉了编译器，我们将会有三个参数张量，并且每个参数张量，都会被按照这样的方式分块，而 `x`、`y`、`z` 就是被分成的块。这一点很重要，我们要意识到，`x`、`y`、`z` 是被分成的块，而不是被分块的张量本身，并且函数体当中的 `x`、`y`、`z` 也都是被分成的块。剩下的就很好理解了（也就剩下 `z = x + y` 一行了，哈哈哈），我们把每一块 `x` 和 `y` 相加，放到了 `z` 中，由于参数张量被分成的每一块都被执行了这样的操作，因此即便对于整体而言，加法也被完成了。

### 矩阵乘法

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

对于矩阵乘法来说，我们也有三个参数张量，但是分块的方式肯定比向量加法要复杂一些。我们将三个矩阵分别记作 $A$、$B$、$C$，其中 $A$ 和 $B$ 为输入，$C$ 为输出。其中 $C$ 的分块操作很简单，我们只需要按照行和列，将其分成大小为 `(BLOCK_SIZE_M, BLOCK_SIZE_N)` 的块即可，这样只要每个这样的块都算出了结果，整个 $C$ 也就都算出了结果。那么该如何分 $A$ 和 $B$ 呢？答案是再引入一个元参数 `BLOCK_SIZE_K`，这样我们就可以把 $A$ 分成 `(BLOCK_SIZE_M, BLOCK_SIZE_K)` 大小的块，把 $B$ 分成 `(BLOCK_SIZE_K, BLOCK_SIZE_N)` 的块。但是对于矩阵乘法，$A$ 和 $B$ 并不是块块对应，而是需要对应 $A$ 的每一行和 $B$ 的每一列，所以我们还需要继续 `tile`，把 $A$ 和 $B$ 进一步分成以行为单位和以列为单位的块。到目前为止，我们有了一堆 $A$ 的行块和 $B$ 的列块，但是对于每一个 $A$ 的行块，我们都要对应 $B$ 的每一个列块。这个时候，我们就需要进行 `expand` 了，我们把 $A$ 的行块沿着列 `expand` 成 $C$ 的列数那么多列，把 $B$ 的列块沿着行 `expand` 成 $C$ 的行数那么多行。这样，我们就成功地将 $A$、$B$、$C$ 三者都分好了块，并且对于每一个 $C$ 的块，我们都有对应好的 $A$ 的行块和 $B$ 的列块。

对应好了分块，后续的部分就简单多了。在函数体当中，我们定义了一个 `accumulator`，用于累加中间结果，之后就遍历了对应好的 $A$ 的行块和 $B$ 的列块，并且把他们相乘的结果累加到了 `accumulator` 当中，最后再将 `accumulator` 放到了对应的 $C$ 的分块当中。由于参数张量被分成的每一块都被执行了这样的操作，因此即便对于整体而言，乘法也被完成了。
