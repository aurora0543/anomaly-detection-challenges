"""核心测量原语 —— 计时、统计汇总、bootstrap 置信区间、预热循环（Handbook §2.5）。

torch 可选：有 CUDA 时用 torch.cuda.Event 精确计时并在前后 synchronize；
否则回退到 time.perf_counter。numpy 可选：缺失时用 statistics 回退。
"""
from __future__ import annotations
import time
from typing import Callable, List, Sequence, Tuple

try:
    import numpy as _np
except Exception:
    _np = None

try:
    import torch as _torch
    _CUDA = _torch.cuda.is_available()
except Exception:
    _torch = None
    _CUDA = False


# --------------------------------------------------------------------------- #
# 计时
# --------------------------------------------------------------------------- #
class Timer:
    """上下文管理器，返回毫秒。GPU 段优先用 cuda.Event。

    用法:
        t = Timer()
        with t:
            model(x)
        ms = t.ms
    """

    def __init__(self, use_cuda: bool | None = None, sync: bool = True):
        self.use_cuda = _CUDA if use_cuda is None else (use_cuda and _CUDA)
        self.sync = sync
        self.ms = 0.0

    def __enter__(self):
        if self.use_cuda:
            self._s = _torch.cuda.Event(enable_timing=True)
            self._e = _torch.cuda.Event(enable_timing=True)
            if self.sync:
                _torch.cuda.synchronize()
            self._s.record()
        else:
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.use_cuda:
            self._e.record()
            _torch.cuda.synchronize()
            self.ms = self._s.elapsed_time(self._e)
        else:
            self.ms = (time.perf_counter() - self._t0) * 1000.0
        return False


def run_timed(fn: Callable[[], None], warmup: int, iters: int,
              use_cuda: bool | None = None) -> List[float]:
    """预热 warmup 次后，正式测量 iters 次单次调用耗时（毫秒）列表。

    fn 是无参可调用，内部执行一次推理（模型与输入在闭包里绑定）。
    """
    for _ in range(warmup):
        fn()
    lat = []
    for _ in range(iters):
        t = Timer(use_cuda=use_cuda)
        with t:
            fn()
        lat.append(t.ms)
    return lat


# --------------------------------------------------------------------------- #
# 统计
# --------------------------------------------------------------------------- #
def _percentile(xs: Sequence[float], q: float) -> float:
    if _np is not None:
        return float(_np.percentile(xs, q))
    s = sorted(xs)
    if not s:
        return float("nan")
    k = (len(s) - 1) * (q / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(latencies_ms: Sequence[float],
              percentiles: Sequence[int] = (50, 95, 99)) -> dict:
    """把一组延迟样本汇总为 mean/std/pXX/max/fps。"""
    xs = list(latencies_ms)
    if not xs:
        return {}
    if _np is not None:
        arr = _np.asarray(xs, dtype=float)
        mean, std = float(arr.mean()), float(arr.std(ddof=1)) if len(xs) > 1 else 0.0
    else:
        import statistics
        mean = statistics.fmean(xs)
        std = statistics.stdev(xs) if len(xs) > 1 else 0.0
    out = {"mean": mean, "std": std, "max": max(xs), "n": len(xs)}
    for q in percentiles:
        out[f"p{q}"] = _percentile(xs, q)
    out["fps"] = 1000.0 / mean if mean > 0 else float("nan")
    return out


def bootstrap_ci(samples: Sequence[float], n_boot: int = 1000,
                 ci: float = 0.95, stat: Callable[[Sequence[float]], float] | None = None
                 ) -> Tuple[float, float]:
    """对给定统计量做自助法置信区间。默认统计量为均值。"""
    xs = list(samples)
    if len(xs) < 2:
        v = xs[0] if xs else float("nan")
        return (v, v)
    if _np is None:
        raise RuntimeError("bootstrap_ci 需要 numpy")
    stat = stat or (lambda a: float(_np.mean(a)))
    arr = _np.asarray(xs, dtype=float)
    rng = _np.random.default_rng(0)
    boots = [stat(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    lo = float(_np.percentile(boots, (1 - ci) / 2 * 100))
    hi = float(_np.percentile(boots, (1 + ci) / 2 * 100))
    return (lo, hi)


def ci_excludes_zero(lo: float, hi: float) -> bool:
    """差异显著性的最简判据：CI 不跨越 0。"""
    return (lo > 0 and hi > 0) or (lo < 0 and hi < 0)
