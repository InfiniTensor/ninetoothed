import torch

import ntops
from ntops.torch.utils import _cached_make


def acosh(input, *, out=None):
    """
    计算反双曲余弦函数 acosh(x) = ln(x + sqrt(x² - 1))
    
    参数:
    input: 输入张量
    out: 可选的输出张量，如果提供，结果将存储在此张量中
    
    返回:
    计算acosh后的张量
    
    数值稳定性:
    - 对于x < 1的值，返回NaN
    - 对于x = 1的值，返回0
    - 对于接近1的值，使用数值稳定的计算方式
    - 支持float16精度，内部使用float32进行中间计算
    """
    # 确定输出数据类型
    tensor_dtype = out.dtype if out is not None else input.dtype
    
    # 创建临时输出张量
    temp_out = torch.empty_like(input, dtype=tensor_dtype, device=input.device)
    
    # 设置块大小，优化GPU/TPU性能
    block_size = 256
    
    # 缓存并获取acosh内核
    # 注意：acosh是element-wise操作，不需要指定归约维度
    # 使用input.ndim作为虚拟维度参数，保持接口一致性
    kernel = _cached_make(
        ntops.kernels.acosh.premake, 
        input.ndim,  # ndim
        0,           # dummy dim (element-wise operation doesn't reduce any dimension)
        dtype=input.dtype,
        block_size=block_size
    )
    
    # 执行内核计算
    kernel(input, temp_out)
    
    # 处理out参数
    if out is not None:
        # 确保out张量的形状和数据类型与结果匹配
        if out.shape != temp_out.shape:
            raise RuntimeError(f"Expected out tensor to have shape {temp_out.shape}, but got {out.shape}")
        if out.dtype != temp_out.dtype:
            raise RuntimeError(f"Expected out tensor to have dtype {temp_out.dtype}, but got {out.dtype}")
        
        # 复制结果到out张量
        out.copy_(temp_out)
        return out
    
    return temp_out


# 为PyTorch张量添加方法
def _acosh_tensor_method(self, *, out=None):
    return acosh(self, out=out)

# 注册到PyTorch Tensor
torch.Tensor.acosh = _acosh_tensor_method


# 为完整性和兼容性提供别名
def arcosh(input, *, out=None):
    """acosh的别名，为了与某些数学库保持一致"""
    return acosh(input, out=out)

def arccosh(input, *, out=None):
    """acosh的别名，为了与NumPy等库保持一致"""
    return acosh(input, out=out)