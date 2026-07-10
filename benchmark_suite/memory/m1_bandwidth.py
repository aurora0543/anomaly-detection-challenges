"""M1 —— HBM 容量与带宽瓶颈：带宽竞争与数据搬运测试（Handbook §3 / 假设 H7）。

目的：验证高速图像采集写入与推理读取共享总线时，总线带宽是否被采集流封顶。
判据（prior）：高分辨率 + 多工位并发采集下 总线带宽利用率近饱和、有效吞吐下降 → H7 支持。

关键场景：多台线阵相机（多验布机）持续以目标 FPS 经 PCIe/DMA 写入内存，与推理读写竞争同一总线。
单路单次推理里"计算"往往主导、搬运占比很小；真正的带宽瓶颈出现在 **高分辨率 × 多工位并发采集**。

执行模型（§2.8）：
  - server（真实）：resident vs e2e 吞吐差、torch.profiler H2D/D2H、nvidia-smi dmon 采带宽。
  - local（可行性）：用总线带宽/采集流量模型合成利用率，演示高分辨率+多工位下的饱和。

可单测核心逻辑：bus_utilization / throughput_drop_pct / judge_h7。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from common.registry import load_registry
from common.runtime import Runtime, load_runtime
from common.models import get_adapter
from common.result import build_result, write_result, validate_result

RESOLUTIONS = [256, 512, 1024, 2048]
_BUS_GBS = 16.0                  # 共享总线（PCIe/边缘内存）有效带宽 GB/s
_ACQ_FPS = 60                    # 每工位采集帧率
_N_STREAMS = 16                  # 并发工位数（多验布机）


@dataclass
class M1Config:
    resolutions: List[int] = field(default_factory=lambda: list(RESOLUTIONS))
    acq_fps: int = _ACQ_FPS
    n_streams: int = _N_STREAMS
    bus_gbs: float = _BUS_GBS
    saturation_tol: float = 0.5      # prior：带宽利用率 >50% 视为进入瓶颈区


# --------------------------------------------------------------------------- #
def bus_utilization(res: int, acq_fps: int, n_streams: int, bus_gbs: float) -> float:
    """多工位采集流占共享总线带宽的比例。"""
    frame_bytes = res * res * 3
    demand_gbs = n_streams * frame_bytes * acq_fps / 1e9      # GB/s
    return demand_gbs / bus_gbs


def throughput_drop_pct(util: float) -> float:
    """带宽饱和后，有效吞吐相对无竞争的下降比例（util≥1 时线性受限）。"""
    return max(0.0, 1.0 - 1.0 / util) if util > 1 else 0.0


def judge_h7(util_hi: float, cfg: M1Config) -> Tuple[str, str]:
    ev = f"高分辨率×{cfg.n_streams}工位 总线利用率 {util_hi*100:.0f}%"
    if util_hi != util_hi:
        return "unfilled", "无带宽数据"
    if util_hi > cfg.saturation_tol:
        return "supported", ev + "（采集流封顶总线，竞争显著）"
    return "not_supported", ev


# --------------------------------------------------------------------------- #
# 真实带宽测量：N 个并发线程持续把 res×res×3 的图像搬进 tensor/设备内存，
# 统计单位时间真实能搬运的字节数（GB/s），作为"总线/内存带宽竞争"的真实代理指标。
# 有 CUDA 时这就是真实 H2D 拷贝带宽；无 GPU（如本机）时测的是 CPU 侧内存拷贝带宽 ——
# 介质不同，但"多路并发数据搬运是否顶到某条总线的真实极限"这件事本身是真的，不是编的数字。
# --------------------------------------------------------------------------- #
def _measure_real_bandwidth(res: int, rt: Runtime, n_streams: int, acq_fps: int,
                            duration_s: float = 1.0) -> Dict[str, float]:
    import threading
    import time

    torch = rt.optional_import("torch")
    if torch is None:
        raise ImportError("M1 真实带宽测量需要 torch")
    device = rt.resolve_device()
    frame_bytes = res * res * 3 * 4  # float32 tensor

    bytes_moved = [0]
    lock = threading.Lock()

    def worker():
        local_bytes = 0
        deadline = time.perf_counter() + duration_s
        while time.perf_counter() < deadline:
            arr = torch.rand(1, 3, res, res)
            if device == "cuda":
                arr = arr.to(device, non_blocking=False)
                torch.cuda.synchronize()
            else:
                arr = arr.clone()  # CPU 侧真实内存拷贝
            local_bytes += frame_bytes
        with lock:
            bytes_moved[0] += local_bytes

    threads = [threading.Thread(target=worker) for _ in range(n_streams)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall_s = time.perf_counter() - t0

    achieved_gbs = (bytes_moved[0] / wall_s / 1e9) if wall_s > 0 else float("nan")
    demand_gbs = n_streams * frame_bytes * acq_fps / 1e9
    util = (demand_gbs / achieved_gbs) if achieved_gbs > 0 else float("nan")
    return {"bus_util": util, "throughput_drop_pct": throughput_drop_pct(util),
            "demand_gbs": demand_gbs, "achieved_gbs": achieved_gbs}


# --------------------------------------------------------------------------- #
def run(model_id: str | None = None, dataset_id: str = "d.mvtec",
        registry=None, runtime: Runtime | None = None, write: bool | None = None) -> Dict[str, Any]:
    registry = registry or load_registry()
    rt = runtime or load_runtime("local")
    model_id = model_id or "m.yolov8n"
    cfg = M1Config()
    adapter = get_adapter(model_id, registry, rt).load()

    per_res = {}
    if rt.is_measurement and adapter.backend == "real":
        for res in cfg.resolutions:
            per_res[res] = _measure_real_bandwidth(res, rt, cfg.n_streams, cfg.acq_fps)
    else:
        for res in cfg.resolutions:
            util = bus_utilization(res, cfg.acq_fps, cfg.n_streams, cfg.bus_gbs)
            per_res[res] = {"bus_util": util, "throughput_drop_pct": throughput_drop_pct(util),
                            "demand_gbs": util * cfg.bus_gbs}
    lo, hi = per_res[cfg.resolutions[0]], per_res[cfg.resolutions[-1]]
    verdict, evidence = judge_h7(hi["bus_util"], cfg)

    res = build_result(
        test_id="M1", test_name="memory_bandwidth_contention",
        measurement_type="real" if rt.is_measurement else "proxy", hardware_id="hw.edge",
        config={"model_id": model_id, "dataset_id": dataset_id, "resolutions": cfg.resolutions,
                "acq_fps": cfg.acq_fps, "n_streams": cfg.n_streams, "bus_gbs": cfg.bus_gbs,
                "mode": rt.name},
        metrics={"per_resolution": per_res,
                 "summary_lines": [
                     f"256px 总线利用率 {lo['bus_util']*100:.0f}% → 2048px {hi['bus_util']*100:.0f}%"
                     f"（{cfg.n_streams} 工位×{cfg.acq_fps}FPS，总线 {cfg.bus_gbs:.0f}GB/s）",
                     f"2048px 饱和导致有效吞吐下降约 {hi['throughput_drop_pct']*100:.0f}%",
                 ]},
        hypothesis_id="H7", verdict=verdict, evidence=evidence,
        notes=("local 可行性：采集流量/总线带宽合成模型，非真实采样。" if rt.is_feasibility
              else "server 真实：N 线程并发真实搬运图像张量测得的实际吞吐（GB/s），"
                   "CUDA 环境下为真实 H2D 拷贝带宽，无 GPU 时为 CPU 侧内存拷贝带宽代理。"),
    )
    if validate_result(res):
        raise RuntimeError("result 不合规")
    if (rt.is_measurement if write is None else write):
        res["_written_to"] = str(write_result(res))
    return res


if __name__ == "__main__":
    import json
    r = run()
    print(json.dumps({"verdict": r["hypothesis"]}, indent=2, ensure_ascii=False))
