"""问题 5.1:per-tensor scale 与 outlier。

构造一个张量:一万个元素均匀分布在 [-1, 1],外加一个 3000 的
outlier。按 per-tensor 方式量化到 E4M3(scale = amax / 448,cast 用
torch.float8_e4m3fn),反量化后测逐点相对误差,填题面的表并回答三问。

跑法:
    uv run python kernels/quant_outlier.py
输出直接用于报告,没有自动判测。
"""

import torch

E4M3_MAX = 448.0


def build_tensor(n: int = 10000, outlier: float = 3000.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    x = torch.rand(n, generator=g) * 2 - 1
    return torch.cat([x, torch.tensor([outlier])])


def quant_dequant_per_tensor(x: torch.Tensor) -> torch.Tensor:
    """per-tensor E4M3 量化再反量化。

    算 scale = amax / 448;除 scale 后 cast 到
    torch.float8_e4m3fn;cast 回 float 再乘 scale。
    """
    scale = x.abs().amax() / E4M3_MAX
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    return (x / scale).to(torch.float8_e4m3fn).float() * scale


def rel_err_at(x: torch.Tensor, y: torch.Tensor, value: float) -> float:
    """取 x 中最接近 value 的元素,返回该点的相对误差。

    零输入重建为零时误差记为零，否则记为无穷大。
    """
    idx = (x - value).abs().argmin()
    error = (y.flatten()[idx] - x.flatten()[idx]).abs().item()
    magnitude = x.flatten()[idx].abs().item()
    return error / magnitude if magnitude else (0.0 if error == 0 else float("inf"))


def quant_dequant_per_block(x: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """沿一维输入划分连续的 block，尾部不足一组时单独计算 scale。"""
    return torch.cat([quant_dequant_per_tensor(block)
                      for block in x.split(block_size)])


def print_stats(label: str, x: torch.Tensor, y: torch.Tensor) -> None:
    nonzero = x != 0
    rel = (y[nonzero] - x[nonzero]).abs() / x[nonzero].abs()
    print(f"  {label}: n={x.numel()}, mean_rel={rel.mean().item():.6e}, "
          f"max_abs={(y - x).abs().max().item():.6e}, "
          f"nonzero_to_zero={((y == 0) & nonzero).sum().item()}")


def main() -> None:
    x = build_tensor()
    y = quant_dequant_per_tensor(x)
    print("含 outlier:")
    for v in (0.5, 0.1, 0.01, 0.005, 3000.0):
        print(f"  x≈{v:<8} rel_err={rel_err_at(x, y, v):.3e}")
    clean = x[:-1]
    y_clean = quant_dequant_per_tensor(clean)
    before = rel_err_at(x, y, 0.5)
    after = rel_err_at(clean, y_clean, 0.5)
    print(f"(a) 去掉 outlier: rel_err(0.5)={after:.6e}, "
          f"含/不含误差比={before / after:.6f}")
    for label, values, restored in (("含 outlier", x, y),
                                     ("不含 outlier", clean, y_clean)):
        scale = values.abs().amax() / E4M3_MAX
        threshold = scale * 2.0**-10
        print(f"(b) {label}: scale={scale.item():.9e}, "
              f"zero_threshold={threshold.item():.9e}")
        # E4M3 的最小 subnormal 为 2^-9，中点按 RN-even 舍入到零。
        probes = threshold * torch.tensor([0.999, 1.0, 1.001])
        decoded = (probes / scale).to(torch.float8_e4m3fn).float() * scale
        print(f"  threshold * [0.999, 1, 1.001] -> {decoded.tolist()}")
        print_stats(label, values, restored)

    y_block = quant_dequant_per_block(x)
    outlier_start = ((x.numel() - 1) // 128) * 128
    print(f"(c) block_size=128, outlier_block_start={outlier_start}, "
          f"outlier_block_length={x.numel() - outlier_start}")
    for label, sl in (("普通 blocks", slice(0, outlier_start)),
                      ("outlier block 的普通元素", slice(outlier_start, -1))):
        print_stats(label + " / per-tensor", x[sl], y[sl])
        print_stats(label + " / per-block", x[sl], y_block[sl])
    print(f"  outlier rel_err={rel_err_at(x, y_block, 3000.0):.6e}")


if __name__ == "__main__":
    main()
