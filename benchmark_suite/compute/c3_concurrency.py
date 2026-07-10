"""C3 —— 扩展性瓶颈：多实例并发压力测试（Handbook §3 / 假设 H4）。

目的：量化单卡承载的并发检测路数上限，验证异常检测模型并发承载是否低于监督模型。
判据（prior）：异常检测组 单卡最大路数(延迟≤节拍) 显著低于 监督组 → H4 支持。

执行模型（§2.8）：
  - server（真实）：启动 N 个推理实例，稳态后统计每路延迟/总吞吐/GPU·CPU 利用率。
  - local（可行性）：用并发争用模型合成 latency(N)，演示异常/监督承载差异。

可单测核心逻辑：max_streams_under_budget / saturation_point / judge_h4。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from common.registry import load_registry
from common.runtime import Runtime, load_runtime
from common.models import get_adapter
from common.result import build_result, write_result, validate_result

CONCURRENCY = [1, 2, 4, 8, 16, 32]
LATENCY_BUDGET_MS = 33.0        # 节拍（实例 ≥30FPS → ≤33ms）

# 并发合成模型：单路基延迟 + 并行容量 k（超过 k 后延迟线性上升）
_BASE_LAT_MS = {"conv": 8.0, "feature_embed": 25.0, "attention": 40.0}
_PARALLEL_K = {"conv": 8, "feature_embed": 3, "attention": 2}   # 异常/注意力更早饱和


@dataclass
class C3Config:
    concurrency: List[int] = field(default_factory=lambda: list(CONCURRENCY))
    latency_budget_ms: float = LATENCY_BUDGET_MS
    duration_s_per_level: float = 2.0   # 每个并发档位真实测量的持续时长


# --------------------------------------------------------------------------- #
# 可单测核心逻辑
# --------------------------------------------------------------------------- #
def max_streams_under_budget(concurrency: List[int], latency_ms: List[float],
                             budget_ms: float) -> int:
    ok = [n for n, l in zip(concurrency, latency_ms) if l <= budget_ms]
    return max(ok) if ok else 0


def saturation_point(concurrency: List[int], total_fps: List[float],
                     rel_gain: float = 0.05) -> int:
    """总吞吐相对增益跌破 rel_gain 的并发数（吞吐饱和点）。"""
    for i in range(1, len(total_fps)):
        prev = total_fps[i - 1]
        if prev > 0 and (total_fps[i] - prev) / prev < rel_gain:
            return concurrency[i]
    return concurrency[-1]


def judge_h4(anomaly_maxstreams: List[int], supervised_maxstreams: List[int]) -> Tuple[str, str]:
    if not anomaly_maxstreams or not supervised_maxstreams:
        return "unfilled", "缺少分组承载数据"
    a = sum(anomaly_maxstreams) / len(anomaly_maxstreams)
    s = sum(supervised_maxstreams) / len(supervised_maxstreams)
    ev = f"异常检测组均值 {a:.1f} 路 vs 监督组 {s:.1f} 路"
    if a < s:
        return "supported", ev + "（异常承载更低）"
    return "not_supported", ev


# --------------------------------------------------------------------------- #
def _synthetic_curve(adapter, cfg: C3Config):
    base = _BASE_LAT_MS.get(adapter.spec.get("paradigm"), 15.0)
    k = _PARALLEL_K.get(adapter.spec.get("paradigm"), 4)
    lat, fps, mem = [], [], []
    for n in cfg.concurrency:
        l = base * max(1.0, n / k)                 # 超过并行容量后线性增长
        lat.append(l)
        fps.append(n * 1000.0 / l)
        mem.append(300 + n * 120)                  # 名义显存 MB
    return lat, fps, mem


def _gpu_mem_used_mb() -> float | None:
    """真实 GPU 显存占用（若有 pynvml + NVIDIA GPU）；否则返回 None（不假装有数据）。"""
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(h)
        pynvml.nvmlShutdown()
        return round(info.used / (1024 ** 2), 1)
    except Exception:
        return None


def _real_curve_threaded(adapter, cfg: C3Config) -> Tuple[List[float], List[float], List[float | None]]:
    """CPU 真实多线程并发压测：每个并发档位起 N 个线程反复调用 adapter.infer()，
    跑 duration_s_per_level 秒，统计该档位下的平均单路延迟与总吞吐。CPU 上多个 OS 线程
    确实能跑到不同的核上，这个并发方式本身是真实的资源争用。
    """
    import threading
    import time

    x = adapter.preprocess(None)
    lat_curve, fps_curve, mem_curve = [], [], []

    for n in cfg.concurrency:
        latencies: List[float] = []
        lock = threading.Lock()

        def worker():
            local_lat = []
            deadline = time.perf_counter() + cfg.duration_s_per_level
            while time.perf_counter() < deadline:
                t0 = time.perf_counter()
                adapter.infer(x)
                local_lat.append((time.perf_counter() - t0) * 1000.0)
            with lock:
                latencies.extend(local_lat)

        threads = [threading.Thread(target=worker) for _ in range(n)]
        wall_start = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall_s = time.perf_counter() - wall_start

        mean_lat = (sum(latencies) / len(latencies)) if latencies else float("nan")
        total_fps = (len(latencies) / wall_s) if wall_s > 0 else float("nan")
        lat_curve.append(mean_lat)
        fps_curve.append(total_fps)
        mem_curve.append(_gpu_mem_used_mb())

    return lat_curve, fps_curve, mem_curve


def _real_curve_batched(adapter, cfg: C3Config, rt: Runtime) -> Tuple[List[float], List[float], List[float | None]]:
    """GPU（mps/cuda）并发压测：单线程、把并发数 N 当成 batch size 连续推理，而不是开
    N 个线程各发一路。

    这不是权宜之计，是真实踩过坑之后确认的正确做法：实测过 PyTorch 的 MPS 后端在多个
    Python 线程同时提交 Metal command buffer 时会直接原生崩溃（`-[_MTLCommandBuffer commit]`
    段错误，退出码 139），不是偶发，是可稳定复现的真实限制，不只是"这里跑不了"那么简单。
    改用批处理后，"单卡支持多少路并发"这件事测的是"batch=N 时的真实吞吐"——这也更贴近生产环境里
    GPU 推理服务实际处理并发请求的方式（动态批处理），不是给每个请求开一个操作系统线程。
    """
    import time

    lat_curve, fps_curve, mem_curve = [], [], []
    for n in cfg.concurrency:
        x = rt.synthetic_image(n, 3, adapter.input_hw, adapter.input_hw)
        n_calls = 0
        deadline_start = time.perf_counter()
        deadline = deadline_start + cfg.duration_s_per_level
        while time.perf_counter() < deadline:
            adapter.infer(x)
            n_calls += 1
        wall_s = time.perf_counter() - deadline_start

        mean_lat = (wall_s / n_calls * 1000.0) if n_calls else float("nan")
        total_fps = (n_calls * n / wall_s) if wall_s > 0 else float("nan")
        lat_curve.append(mean_lat)
        fps_curve.append(total_fps)
        mem_curve.append(_gpu_mem_used_mb())

    return lat_curve, fps_curve, mem_curve


def _real_curve(adapter, cfg: C3Config, rt: Runtime) -> Tuple[List[float], List[float], List[float | None]]:
    device = rt.resolve_device()
    if device == "cpu":
        return _real_curve_threaded(adapter, cfg)
    return _real_curve_batched(adapter, cfg, rt)


def run(model_id: str | None = None, dataset_id: str = "d.sdust",
        registry=None, runtime: Runtime | None = None, write: bool | None = None) -> Dict[str, Any]:
    registry = registry or load_registry()
    rt = runtime or load_runtime("local")
    cfg = C3Config()

    # 分组：监督(检测) vs 异常(特征嵌入)
    sup = [m for m, s in registry.models.items() if s.get("task") == "detection"]
    ano = [m for m, s in registry.models.items() if s.get("paradigm") == "feature_embed"]
    targets = (sup + ano) if model_id is None else [model_id]

    per_model: Dict[str, Any] = {}
    maxstreams: Dict[str, int] = {}
    for mid in targets:
        adapter = get_adapter(mid, registry, rt).load()
        if rt.is_measurement and adapter.backend == "real":
            lat, fps, mem = _real_curve(adapter, cfg, rt)
        else:
            lat, fps, mem = _synthetic_curve(adapter, cfg)
        ms = max_streams_under_budget(cfg.concurrency, lat, cfg.latency_budget_ms)
        maxstreams[mid] = ms
        per_model[mid] = {"task": registry.models[mid].get("task"),
                          "paradigm": registry.models[mid].get("paradigm"),
                          "backend": adapter.backend,
                          "concurrency": cfg.concurrency, "latency_ms": lat,
                          "total_fps": fps, "gpu_mem_mb": mem,
                          "max_streams_under_budget": ms,
                          "saturation_N": saturation_point(cfg.concurrency, fps)}

    verdict, evidence = judge_h4([maxstreams[m] for m in ano if m in maxstreams],
                                 [maxstreams[m] for m in sup if m in maxstreams])

    res = build_result(
        test_id="C3", test_name="multi_instance_concurrency",
        measurement_type="real" if rt.is_measurement else "proxy", hardware_id="hw.cloud",
        config={"concurrency": cfg.concurrency, "latency_budget_ms": cfg.latency_budget_ms,
                "mode": rt.name, "targets": targets},
        metrics={"per_model": per_model, "max_streams": maxstreams,
                 "summary_lines": [f"{m}: 最大 {maxstreams[m]} 路(≤{cfg.latency_budget_ms:.0f}ms)"
                                   for m in maxstreams]},
        hypothesis_id="H4", verdict=verdict, evidence=evidence,
        notes=("local 可行性：并发争用合成模型，非真实并发计时。" if rt.is_feasibility else ""),
    )
    if validate_result(res):
        raise RuntimeError("result 不合规")
    if (rt.is_measurement if write is None else write):
        res["_written_to"] = str(write_result(res))
    return res


if __name__ == "__main__":
    import json
    r = run()
    print(json.dumps({"verdict": r["hypothesis"], "max_streams": r["metrics"]["max_streams"]},
                     indent=2, ensure_ascii=False))
