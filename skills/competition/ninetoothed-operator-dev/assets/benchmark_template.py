"""
九齿算子基准测试模板文件

使用说明：
1. 将此文件复制为新基准测试文件
2. 替换算子创建函数
3. 配置测试参数
4. 运行基准测试

九齿算子使用 arrange-and-apply 范式，需要使用 ninetoothed.make() 构建内核。
"""

import torch
import time
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file))))


def create_benchmark_kernel():
    """
    创建基准测试算子内核

    此函数应替换为实际的算子创建逻辑

    Returns:
        九齿内核函数
    """
    import ninetoothed
    import ninetoothed.language as ntl
    from ninetoothed import Tensor, Symbol

    BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)

    def arrangement(x, y, output, BLOCK_SIZE=128):
        return (x.tile((BLOCK_SIZE,)),
               y.tile((BLOCK_SIZE,)),
               output.tile((BLOCK_SIZE,)))

    def application(x, y, output):
        output = x + y  # noqa: F841

    return ninetoothed.make(arrangement, application,
                          (Tensor(1), Tensor(1), Tensor(1)))


def benchmark_operator(
    kernel_func,
    input_tensors,
    num_warmup=10,
    num_iterations=100,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    基准测试单个九齿算子

    Args:
        kernel_func: 九齿内核函数（需要传入实际创建的内核）
        input_tensors: 输入张量元组
        num_warmup: 热身次数
        num_iterations: 测试次数
        device: 设备类型

    Returns:
        基准测试结果字典
    """
    if device == 'cuda':
        torch.cuda.synchronize()

    for _ in range(num_warmup):
        _ = kernel_func(*input_tensors)

    if device == 'cuda':
        torch.cuda.synchronize()

    times = []
    for _ in range(num_iterations):
        start = time.time()
        _ = kernel_func(*input_tensors)
        if device == 'cuda':
            torch.cuda.synchronize()
        end = time.time()
        times.append(end - start)

    times_np = np.array(times)
    total_elements = sum(t.numel() for t in input_tensors)

    return {
        'device': device,
        'mean_ms': np.mean(times_np) * 1000,
        'median_ms': np.median(times_np) * 1000,
        'std_ms': np.std(times_np) * 1000,
        'min_ms': np.min(times_np) * 1000,
        'max_ms': np.max(times_np) * 1000,
        'throughput_gbs': total_elements * 4 / 1e9 / np.mean(times_np)
    }


def compare_with_pytorch(
    kernel_func,
    torch_func,
    input_tensors,
    num_warmup=10,
    num_iterations=100
):
    """
    对比九齿算子与PyTorch实现

    Args:
        kernel_func: 九齿内核函数
        torch_func: PyTorch等价函数
        input_tensors: 输入张量
        num_warmup: 热身次数
        num_iterations: 测试次数

    Returns:
        对比结果字典
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    nt_result = benchmark_operator(kernel_func, input_tensors, num_warmup, num_iterations, device)

    torch_times = []
    for _ in range(num_warmup):
        _ = torch_func(*input_tensors)
    if device == 'cuda':
        torch.cuda.synchronize()

    for _ in range(num_iterations):
        start = time.time()
        _ = torch_func(*input_tensors)
        if device == 'cuda':
            torch.cuda.synchronize()
        end = time.time()
        torch_times.append(end - start)

    torch_times_np = np.array(torch_times)
    total_elements = sum(t.numel() for t in input_tensors)

    torch_result = {
        'device': device,
        'mean_ms': np.mean(torch_times_np) * 1000,
        'median_ms': np.median(torch_times_np) * 1000,
        'std_ms': np.std(torch_times_np) * 1000,
        'min_ms': np.min(torch_times_np) * 1000,
        'max_ms': np.max(torch_times_np) * 1000,
        'throughput_gbs': total_elements * 4 / 1e9 / np.mean(torch_times_np)
    }

    speedup = torch_result['mean_ms'] / nt_result['mean_ms'] if nt_result['mean_ms'] > 0 else float('inf')

    return {
        'nine_toothed': nt_result,
        'pytorch': torch_result,
        'speedup': speedup
    }


def print_results(results, title="基准测试结果"):
    """
    打印基准测试结果

    Args:
        results: 测试结果字典
        title: 结果标题
    """
    print(f"\n=== {title} ===")
    print(f"设备: {results['device']}")

    print(f"\n平均耗时: {results['mean_ms']:.3f} ms")
    print(f"中位数耗时: {results['median_ms']:.3f} ms")
    print(f"标准差: {results['std_ms']:.3f} ms")
    print(f"最小耗时: {results['min_ms']:.3f} ms")
    print(f"最大耗时: {results['max_ms']:.3f} ms")
    print(f"吞吐量: {results['throughput_gbs']:.2f} GB/s")


def print_comparison(results, title="性能对比"):
    """
    打印对比结果

    Args:
        results: 对比结果字典
        title: 结果标题
    """
    print(f"\n=== {title} ===")
    print(f"设备: {results['nine_toothed']['device']}")

    print("\n九齿算子:")
    print(f"  平均耗时: {results['nine_toothed']['mean_ms']:.3f} ms")
    print(f"  吞吐量: {results['nine_toothed']['throughput_gbs']:.2f} GB/s")

    print("\nPyTorch算子:")
    print(f"  平均耗时: {results['pytorch']['mean_ms']:.3f} ms")
    print(f"  吞吐量: {results['pytorch']['throughput_gbs']:.2f} GB/s")

    print(f"\n加速比: {'{:.2f}x'.format(results['speedup']) if results['speedup'] else 'N/A'}")

    if results['speedup'] and results['speedup'] > 1:
        print("✓ 九齿算子性能更优")
    elif results['speedup'] and results['speedup'] < 1:
        print("✗ PyTorch性能更优，建议优化")
    else:
        print("≈ 性能相当")


def create_add_kernel(BLOCK_SIZE=128):
    """创建 add 算子内核"""
    import ninetoothed
    import ninetoothed.language as ntl
    from ninetoothed import Tensor, Symbol

    BLOCK_SIZE_VAL = Symbol("BLOCK_SIZE_VAL", constexpr=True)

    def arrangement(x, y, output, BLOCK_SIZE_VAL=BLOCK_SIZE_VAL):
        return (x.tile((BLOCK_SIZE_VAL,)),
               y.tile((BLOCK_SIZE_VAL,)),
               output.tile((BLOCK_SIZE_VAL,)))

    def application(x, y, output):
        output = x + y  # noqa: F841

    return ninetoothed.make(arrangement, application,
                          (Tensor(1), Tensor(1), Tensor(1)))


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=== 九齿算子基准测试模板 ===\n")

    test_shapes = [
        (1024,),
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
    ]

    for shape in test_shapes:
        print(f"\n测试形状: {shape}")

        kernel = create_add_kernel()
        a = torch.randn(shape, dtype=torch.float16, device=device)
        b = torch.randn(shape, dtype=torch.float16, device=device)
        c = torch.empty_like(a)

        results = benchmark_operator(kernel, (a, b, c))
        print_results(results, f"Add算子 - 形状 {shape}")

    print("\n" + "=" * 60)
    print("与 PyTorch 对比测试")
    print("=" * 60)

    for shape in test_shapes:
        print(f"\n对比形状: {shape}")

        kernel = create_add_kernel()
        a = torch.randn(shape, dtype=torch.float16, device=device)
        b = torch.randn(shape, dtype=torch.float16, device=device)
        c = torch.empty_like(a)

        def torch_add(a, b, c):
            c.copy_(a + b)

        comparison = compare_with_pytorch(kernel, torch_add, (a, b, c))
        print_comparison(comparison, f"Add vs PyTorch - 形状 {shape}")
