import torch

import ntops
from ntops.torch.utils import _cached_make


def atan(input, *, out=None):
    """
    计算反正切函数 atan(x)
    
    参数:
    input: 输入张量
    out: 可选的输出张量
    
    返回:
    计算 atan 后的张量
    
    数值特性:
    - 定义域: (-inf, +inf)
    - 值域: (-pi/2, +pi/2)
    - 支持 float16 (内部使用 float32 计算)
    """
    # 确定输出数据类型
    tensor_dtype = out.dtype if out is not None else input.dtype
    
    # 创建临时输出张量
    temp_out = torch.empty_like(input, dtype=tensor_dtype, device=input.device)
    
    # 设置块大小
    block_size = 256
    
    # 缓存并获取 atan 内核
    kernel = _cached_make(
        ntops.kernels.atan.premake, 
        input.ndim,  # ndim
        0,           # dummy dim
        dtype=input.dtype,
        block_size=block_size
    )
    
    # 执行内核计算
    kernel(input, temp_out)
    
    # 处理 out 参数
    if out is not None:
        if out.shape != temp_out.shape:
            raise RuntimeError(f"Expected out tensor to have shape {temp_out.shape}, but got {out.shape}")
        if out.dtype != temp_out.dtype:
            raise RuntimeError(f"Expected out tensor to have dtype {temp_out.dtype}, but got {out.dtype}")
        
        out.copy_(temp_out)
        return out
    
    return temp_out


# 为 PyTorch 张量添加方法
def _atan_tensor_method(self, *, out=None):
    return atan(self, out=out)

# 注册到 PyTorch Tensor
torch.Tensor.atan = _atan_tensor_method


# 别名支持 (兼容 NumPy 命名习惯)
def arctan(input, *, out=None):
    """atan 的别名"""
    return atan(input, out=out)