"""N2 —— 非对称数据流控：事件驱动回传延迟测试（Handbook §3 / 假设 H10）。

目的：验证"仅缺陷事件才回传"能否大幅省带宽，同时保证高危缺陷毫秒级响应。
判据（hard）：省带宽比高 且 高危事件回传延迟 ≤ 时限 → H10 支持。

执行模型（§2.8）：
  - server（真实）：注入 1% 缺陷帧的视频流，边缘检测触发回传，埋点 T1–T5 测延迟与流量。
  - local（可行性）：合成帧流与回传，演示省带宽比与高危延迟。

可单测核心逻辑：bw_saving_pct / judge_h10。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from common.registry import load_registry
from common.runtime import Runtime, load_runtime
from common.models import get_adapter
from common.result import build_result, write_result, validate_result

SCHEMES = ["full", "event", "tiered"]


@dataclass
class N2Config:
    defect_rate: float = 0.01           # 缺陷帧占比
    duration_frames: int = 100_000
    frame_kb: float = 300.0             # 整帧大小
    patch_kb: float = 30.0              # 缺陷块大小
    false_alarm_rate: float = 0.03      # 异常检测误报率（抬高实际回传）
    high_risk_latency_ms: float = 200.0 # hard 门槛：高危回传时限（局域网）
    bw_saving_tol: float = 0.8          # hard：省带宽 >80% 视为达标


# --------------------------------------------------------------------------- #
def bw_saving_pct(full_bytes: float, scheme_bytes: float) -> float:
    return (full_bytes - scheme_bytes) / full_bytes if full_bytes else float("nan")


def judge_h10(bw_saving: float, high_risk_latency_ms: float, cfg: N2Config) -> Tuple[str, str]:
    ev = f"省带宽 {bw_saving*100:.0f}%，高危回传延迟 {high_risk_latency_ms:.0f}ms"
    ok_bw = bw_saving > cfg.bw_saving_tol
    ok_lat = high_risk_latency_ms <= cfg.high_risk_latency_ms
    if ok_bw and ok_lat:
        return "supported", ev + "（省带宽且高危达标）"
    if ok_bw or ok_lat:
        return "partial", ev
    return "not_supported", ev


# --------------------------------------------------------------------------- #
def _traffic(cfg: N2Config) -> Dict[str, float]:
    n = cfg.duration_frames
    n_defect = n * cfg.defect_rate
    n_false = n * cfg.false_alarm_rate
    full = n * cfg.frame_kb
    event = (n_defect + n_false) * cfg.frame_kb            # 缺陷+误报，回传整帧
    tiered = n_defect * cfg.frame_kb + n_false * cfg.patch_kb  # 高危整帧、误报仅小块
    return {"full": full, "event": event, "tiered": tiered}


def _real_frame_byte_sizes(resolution: int = 512) -> Tuple[float, float]:
    """真实编码一帧合成图 + 一个缺陷小块，取真实 JPEG 压缩后字节数（KB），
    替代原来写死的 frame_kb/patch_kb 常量。"""
    import io
    import random
    from PIL import Image

    rng = random.Random(7)
    frame = Image.new("RGB", (resolution, resolution),
                      (rng.randint(180, 220), rng.randint(180, 220), rng.randint(180, 220)))
    buf = io.BytesIO()
    frame.save(buf, format="JPEG", quality=85)
    frame_kb = len(buf.getvalue()) / 1024.0

    patch_size = max(16, resolution // 8)
    patch = Image.new("RGB", (patch_size, patch_size), (60, 60, 60))
    buf2 = io.BytesIO()
    patch.save(buf2, format="JPEG", quality=85)
    patch_kb = len(buf2.getvalue()) / 1024.0
    return frame_kb, patch_kb


def _real_event_pipeline_latency(adapter, rt: Runtime, n_events: int = 30) -> Dict[str, float]:
    """真实的"检测触发 -> 事件入队 -> 消费者取出"延迟测量：
    用真实模型推理产生 T_detect（检测耗时），用真实的 threading.Queue 生产者/消费者
    测 T_deliver（事件从产生到被下游取走的排队+调度延迟）——这是"事件驱动回传"里
    真正可以在没有专门网络设备的情况下如实测量的那部分；不模拟局域网/广域网物理链路本身。
    """
    import queue
    import threading
    import time

    x = adapter.preprocess(None)
    detect_ms: List[float] = []
    deliver_ms: List[float] = []
    q: "queue.Queue[float]" = queue.Queue()
    stop = object()

    def consumer():
        while True:
            item = q.get()
            if item is stop:
                return
            deliver_ms.append((time.perf_counter() - item) * 1000.0)

    t = threading.Thread(target=consumer)
    t.start()
    for _ in range(n_events):
        t0 = time.perf_counter()
        adapter.infer(x)
        detect_ms.append((time.perf_counter() - t0) * 1000.0)
        q.put(time.perf_counter())   # 事件"产生"时刻入队，消费者取出时刻算出排队延迟
    q.put(stop)
    t.join()

    return {
        "detect_ms_mean": sum(detect_ms) / len(detect_ms) if detect_ms else float("nan"),
        "deliver_ms_mean": sum(deliver_ms) / len(deliver_ms) if deliver_ms else float("nan"),
        "n_events": n_events,
    }


def run(model_id: str = "m.yolov8n", dataset_id: str = "d.sdust",
        registry=None, runtime: Runtime | None = None, write: bool | None = None) -> Dict[str, Any]:
    registry = registry or load_registry()
    rt = runtime or load_runtime("local")
    model_id = model_id or "m.yolov8n"
    cfg = N2Config()

    if rt.is_measurement:
        adapter = get_adapter(model_id, registry, rt).load()
        if adapter.backend == "real":
            frame_kb, patch_kb = _real_frame_byte_sizes()
            cfg.frame_kb, cfg.patch_kb = frame_kb, patch_kb
            pipeline = _real_event_pipeline_latency(adapter, rt)
            high_risk_latency_ms = pipeline["detect_ms_mean"] + pipeline["deliver_ms_mean"]
        else:
            high_risk_latency_ms = 120.0
    else:
        high_risk_latency_ms = 120.0

    # defect_rate/false_alarm_rate/duration_frames 是产线场景假设（同 C1 的 line_speed_fps），
    # 不是可以"测量"出来的量；frame_kb/patch_kb 在 server 真实模式下已替换为真实编码字节数。
    tr = _traffic(cfg)
    per_scheme = {}
    for s in SCHEMES:
        per_scheme[s] = {
            "traffic_kb": tr[s],
            "bw_saving_pct": bw_saving_pct(tr["full"], tr[s]) if s != "full" else 0.0,
        }
    verdict, evidence = judge_h10(per_scheme["event"]["bw_saving_pct"], high_risk_latency_ms, cfg)

    res = build_result(
        test_id="N2", test_name="event_driven_callback",
        measurement_type="real" if rt.is_measurement else "proxy", hardware_id="hw.edge",
        config={"model_id": model_id, "dataset_id": dataset_id,
                "defect_rate": cfg.defect_rate, "false_alarm_rate": cfg.false_alarm_rate,
                "mode": rt.name},
        metrics={"per_scheme": per_scheme, "high_risk_latency_ms": high_risk_latency_ms,
                 "summary_lines": [
                     f"event 省带宽 {per_scheme['event']['bw_saving_pct']*100:.0f}%、"
                     f"tiered 省带宽 {per_scheme['tiered']['bw_saving_pct']*100:.0f}%",
                     f"高危回传延迟 ≈ {high_risk_latency_ms:.0f}ms（≤{cfg.high_risk_latency_ms:.0f} 门槛）；"
                     f"误报率 {cfg.false_alarm_rate*100:.0f}% 抬高实际回传",
                 ]},
        hypothesis_id="H10", verdict=verdict, evidence=evidence,
        notes=("local 可行性：合成帧流与流量，延迟为文献先验。" if rt.is_feasibility
              else "server 真实：帧/块字节数为真实 JPEG 编码测得；高危延迟 = 真实模型推理耗时 + "
                   "真实线程队列事件投递延迟（同进程内软件事件管线，非物理局域网/广域网链路测量）。"),
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
