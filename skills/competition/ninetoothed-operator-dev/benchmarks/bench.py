"""
九齿算子 benchmark 包装器。

遵循 ninetoothed-examples/bench.py 的设计模式，基于 triton.testing 原语：
  - triton.testing.Benchmark: 声明扫描空间
  - triton.testing.perf_report: 装饰器，串联 Benchmark → 执行函数
  - triton.testing.do_bench: 单次计时调用

用法：
    from benchmarks.bench import benchmark
    benchmark(impls={"ninetoothed": kernel_func, "torch": torch_func},
              make_inputs=lambda size: ((x, y, out), {}),
              x_names=["size"], x_vals=[2**i for i in range(10, 20)],
              save_path="output", plot_name="my_op")
"""
import os
import torch
import triton
import triton.testing


def benchmark(
    impls,
    make_inputs,
    x_names,
    x_vals,
    plot_name="benchmark",
    x_log=True,
    ylabel="ms",
    save_path=None,
    benchmark_args=None,
    tolerances=None,
    skip_correctness=False,
):
    """
    统一的 benchmark 入口。

    Args:
        impls:       OrderedDict[str, callable] — {"ninetoothed": fn, "torch": fn, ...}
        make_inputs: callable(sweep_point) -> ((args,), {kwargs}) — 按扫描点生成输入
        x_names:     list[str] — 扫描参数的名称
        x_vals:      list — 扫描参数的取值序列
        plot_name:   str — 图表标题
        x_log:       bool — x 轴是否对数
        ylabel:      str — y 轴标签
        save_path:   str|None — 图片保存目录
        benchmark_args: dict|None — 传递给 Benchmark 的额外参数
        tolerances:  dict|None — {"provider": {"atol": ..., "rtol": ...}}
    """
    if benchmark_args is None:
        benchmark_args = {}

    if tolerances is None:
        tolerances = {}

    # --- 内部 bench 函数（被 perf_report 装饰） ---
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=x_names,
            x_vals=x_vals,
            line_arg="provider",
            line_vals=list(impls.keys()),
            line_names=list(impls.keys()),
            styles=[("blue", "-"), ("green", "-"), ("orange", "-")],
            ylabel=ylabel,
            plot_name=plot_name,
            args=benchmark_args,
            x_log=x_log,
        )
    )
    def _bench(provider, **x_kwargs):
        # 生成输入
        inputs, kwargs = make_inputs(**x_kwargs)
        fn = impls[provider]

        # 正确性检验（除第一个 provider 外，都与第一个对比）
        # 注意：ninetoothed kernel 在 Tensor 上原地写结果（返回 None），
        # 所以比较输出张量的内容而非函数返回值
        if not skip_correctness and provider != list(impls.keys())[0]:
            ref_fn = impls[list(impls.keys())[0]]
            # 克隆输出张量，避免 ref_fn 和 fn 共享同一个 tensor
            ref_in = tuple(t.clone() if isinstance(t, torch.Tensor) and t.dtype != torch.int64
                          else t for t in inputs)
            ref_fn(*ref_in, **kwargs)
            fn(*inputs, **kwargs)
            tol = tolerances.get(provider, {"atol": 1e-2, "rtol": 1e-2})
            # 比较所有输出张量
            for a, b in zip(ref_in, inputs):
                if isinstance(a, torch.Tensor) and a.dtype != torch.int64:
                    torch.testing.assert_close(b, a, **tol)

        # 计时
        quantiles = [0.5, 0.2, 0.8]
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: fn(*inputs, **kwargs),
            quantiles=quantiles,
        )
        return ms, max_ms, min_ms

    # 执行
    _bench.run(print_data=True, save_path=save_path, show_plots=False)


def make_inputs_from_shapes(create_fn, dtype=torch.float16):
    """
    便捷工厂：根据 shape 生成 (inputs, outputs) 元组。

    用法:
        make_inputs_from_shapes(
            lambda shape: (torch.randn(shape, dtype=torch.float16),)
        )
    """
    def factory(**x_kwargs):
        shapes = tuple(
            x_kwargs[name] if isinstance(x_kwargs[name], tuple) else (x_kwargs[name],)
            for name in x_kwargs
        )
        tensors = create_fn(*shapes, dtype=dtype)
        if isinstance(tensors, tuple):
            return tensors, {}
        return (tensors,), {}
    return factory
