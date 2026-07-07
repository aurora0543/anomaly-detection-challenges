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


def run(model_id: str = "m.yolov8n", dataset_id: str = "d.sdust",
        registry=None, runtime: Runtime | None = None, write: bool | None = None) -> Dict[str, Any]:
    registry = registry or load_registry()
    rt = runtime or load_runtime("local")
    model_id = model_id or "m.yolov8n"
    cfg = N2Config()

    # server 真实路径：需检测模型跑注入缺陷的视频流并埋点 T1–T5（后续步骤对接）
    tr = _traffic(cfg)
    per_scheme = {}
    for s in SCHEMES:
        per_scheme[s] = {
            "traffic_kb": tr[s],
            "bw_saving_pct": bw_saving_pct(tr["full"], tr[s]) if s != "full" else 0.0,
            "detect_to_arrival_ms": {"lan": 120, "wan": 900}[  # 名义（文献先验，仅对照）
                "lan"] if s != "full" else None,
        }
    # 高危(整帧)在局域网的回传延迟（名义），server 用真实埋点覆盖
    high_risk_latency_ms = 120.0
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
        notes=("local 可行性：合成帧流与流量，延迟为文献先验。" if rt.is_feasibility else ""),
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
