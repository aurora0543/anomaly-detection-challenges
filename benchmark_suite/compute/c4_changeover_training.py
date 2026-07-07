"""C4 —— 训练硬件门槛：换产训练成本测试（Handbook §3 / 假设 H3）。

目的：量化"换产"时重训/增量的时间与资源代价，验证边缘能否独立完成换产适配。
判据（mixed）：全量训练耗时（小时级）高 且 边/云训练效率比大 → H3 支持。

场景：A_full 全量 / B_incremental 增量微调 / C_fewshot / D_edge_incremental（无边缘设备标待验证）。

执行模型（§2.8）：
  - server（真实）：ultralytics.train / 仓库 main.py 训到收敛，记录耗时/能耗/峰值显存/最终精度。
  - local（可行性）：合成各场景训练成本，演示"换产延迟高、边缘慢"。

可单测核心逻辑：full_vs_incremental_speedup / judge_h3。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from common.registry import load_registry
from common.runtime import Runtime, load_runtime
from common.models import get_adapter
from common.result import build_result, write_result, validate_result

SCENARIOS = ["A_full", "B_incremental", "C_fewshot", "D_edge_incremental"]
NEW_SAMPLE_SIZES = [50, 200, 1000]

# 合成成本基准（cloud）：全量小时、增量分钟；edge 相对 cloud 的减速倍数
_CLOUD_FULL_H = {"conv": 2.0, "feature_embed": 1.2, "attention": 3.5}
_CLOUD_INCR_MIN = {"conv": 12.0, "feature_embed": 8.0, "attention": 20.0}
_EDGE_SLOWDOWN = 20.0            # 边缘比云端慢约 20x（无风扇低功耗）


@dataclass
class C4Config:
    scenarios: List[str] = field(default_factory=lambda: list(SCENARIOS))
    new_sample_sizes: List[int] = field(default_factory=lambda: list(NEW_SAMPLE_SIZES))
    full_hours_threshold: float = 1.0     # prior：全量≥1h 视为"换产延迟高"
    edge_ratio_threshold: float = 5.0     # prior：边/云≥5x 视为"边缘难独立完成"


# --------------------------------------------------------------------------- #
def full_vs_incremental_speedup(full_h: float, incr_min: float) -> float:
    return (full_h * 60.0) / incr_min if incr_min else float("nan")


def judge_h3(full_h: float, edge_cloud_ratio: float, cfg: C4Config) -> Tuple[str, str]:
    ev = f"全量 {full_h:.1f}h，边/云效率比 {edge_cloud_ratio:.0f}x"
    hi_delay = full_h >= cfg.full_hours_threshold
    edge_limited = edge_cloud_ratio >= cfg.edge_ratio_threshold
    if hi_delay and edge_limited:
        return "supported", ev + "（换产延迟高且边缘受限）"
    if hi_delay or edge_limited:
        return "partial", ev
    return "not_supported", ev


# --------------------------------------------------------------------------- #
def run(model_id: str = "m.yolov8n", dataset_id: str = "d.zju",
        registry=None, runtime: Runtime | None = None, write: bool | None = None) -> Dict[str, Any]:
    registry = registry or load_registry()
    rt = runtime or load_runtime("local")
    model_id = model_id or "m.yolov8n"
    cfg = C4Config()
    adapter = get_adapter(model_id, registry, rt).load()
    pdm = registry.models[model_id].get("paradigm")

    if rt.is_measurement and adapter.backend == "real":
        raise NotImplementedError("C4 真实训练在 server 步骤对接 ultralytics.train / 仓库 main.py（训到收敛）")

    full_h = _CLOUD_FULL_H.get(pdm, 2.0)
    incr_min = _CLOUD_INCR_MIN.get(pdm, 12.0)
    per_scenario = {
        "A_full": {"hw": "cloud", "train_time_h": full_h, "gpu_mem_peak_gb": 12.0, "final_metric": 0.86},
        "B_incremental": {"hw": "cloud", "train_time_min": incr_min, "gpu_mem_peak_gb": 8.0, "final_metric": 0.83},
        "C_fewshot": {"hw": "cloud", "train_time_min": incr_min * 0.4, "final_metric": 0.74},
        "D_edge_incremental": ({"hw": "edge", "train_time_min": incr_min * _EDGE_SLOWDOWN,
                                "final_metric": 0.80}
                               if registry.hardware["hw.edge"]["status"] == "available"
                               else {"hw": "edge", "status": "待验证（无边缘设备）"}),
    }
    speedup = full_vs_incremental_speedup(full_h, incr_min)
    edge_ratio = _EDGE_SLOWDOWN
    verdict, evidence = judge_h3(full_h, edge_ratio, cfg)

    res = build_result(
        test_id="C4", test_name="changeover_training_cost",
        measurement_type="real" if rt.is_measurement else "proxy", hardware_id="hw.cloud",
        config={"model_id": model_id, "dataset_id": dataset_id, "scenarios": cfg.scenarios,
                "new_sample_sizes": cfg.new_sample_sizes, "mode": rt.name},
        metrics={"per_scenario": per_scenario,
                 "full_vs_incremental_speedup": speedup,
                 "edge_vs_cloud_ratio": edge_ratio,
                 "summary_lines": [
                     f"全量 {full_h:.1f}h vs 增量 {incr_min:.0f}min（提速 {speedup:.0f}x）",
                     f"边/云效率比 ≈ {edge_ratio:.0f}x；D 场景 {per_scenario['D_edge_incremental'].get('status','已跑')}",
                 ]},
        hypothesis_id="H3", verdict=verdict, evidence=evidence,
        notes=("local 可行性：合成训练成本，非真实训练。" if rt.is_feasibility else ""),
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
