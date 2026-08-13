"""问题 7.7（压轴）：softmax in TileLang（FROM-SCRATCH）。

contract：
- softmax(x) 接收形状 (M, N) 的 float32 CUDA tensor，返回同形状结果，
  对每一行独立做 softmax；
- kernel 用 TileLang 自己写，一个 block 处理一行（或一小批行）；
- 为了确保数值稳定，要求行内先减最大值，再做 exp 与求和。测试里有一行
  数值巨大的输入，不稳定的实现会得到 inf/nan；
- 行宽 N 任意，可以假设 N <= 4096。TileLang 的 kernel 按形状编译，
  用 make_xxx(M, N) 针对形状生成、在 wrapper 里按形状缓存编译结果
  是常见做法（结构可以参考 7.3、7.4）；
- 归约用 T.reduce_max / T.reduce_sum，逐元素部分用 T.Parallel 加 T.exp；
- fragment 的宽度建议取不小于 N 的 2 的幂（类比 Triton 的
  next_power_of_2），不足的位置补 -inf（T.if_then_else 加 T.infinity），
  否则布局推断可能报 no available layout；
- 通过 pytest tests/test_tilelang_softmax.py 即为完成。

(Optional) 将你的实现和 torch.softmax 比较一下性能（行宽取 256/1024/4096），
Tip: elementwise + 行内归约的 kernel 大概率是带宽瓶颈，可以想想理论上限是多少。
"""

from numpy import dtype
from sympy import N
import torch
import tilelang
import tilelang.language as T


def softmax(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError(f"softmax expects a 2-D tensor, got shape {tuple(x.shape)}")
    if x.dtype != torch.float32:
        raise TypeError(f"softmax expects float32 input, got {x.dtype}")
    if not x.is_cuda:
        raise ValueError("TileLang softmax expects a CUDA tensor")

    M, N = map(int, x.shape)
    if M == 0 or N == 0:
        return torch.softmax(x, dim=-1)

    if not x.is_contiguous():
        x = x.contiguous()

    width = 1 if N <= 1 else 1 << (N - 1).bit_length()
    cache = getattr(softmax, "_compiled", None)
    if cache is None:
        cache = {}
        softmax._compiled = cache

    key = (M, N, x.device)
    kernel = cache.get(key)
    if kernel is None:
        @T.prim_func
        def softmax_kernel(
            A: T.Buffer((M, N), "float32"),
            B: T.Buffer((M, N), "float32"),
        ):
            with T.Kernel(T.ceildiv(M, 1), 1, threads=256) as (bx, _by):
                values = T.alloc_fragment((width,), "float32")
                shifted_exp = T.alloc_fragment((width,), "float32")
                row_max = T.alloc_fragment((1,), "float32")
                row_sum = T.alloc_fragment((1,), "float32")

                for j in T.Parallel(width):
                    values[j] = T.if_then_else(
                        j < N, A[bx, j], -T.infinity("float32")
                    )

                T.reduce_max(values, row_max, dim=0)

                for j in T.Parallel(width):
                    shifted_exp[j] = T.exp(values[j] - row_max[0])

                T.reduce_sum(shifted_exp, row_sum, dim=0)

                for j in T.Parallel(width):
                    if j < N:
                        B[bx, j] = shifted_exp[j] / row_sum[0]

        kernel = tilelang.compile(softmax_kernel, out_idx=[1])
        cache[key] = kernel
    return kernel(x)
