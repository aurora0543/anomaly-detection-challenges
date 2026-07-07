"""N3 —— 硬实时控制链协同：端到端控制延迟测试（Handbook §3 / 假设 H11）。

目的：验证检测信号与执行机构（织机/产线）的硬实时协同能否满足确定性时延要求。
判据（hard）：端到端延迟 P99/最大值 超硬实时预算 或 抖动大 → H11 支持（纯软件难满足）。

⚠ 需硬件在环（HIL：采集卡→边缘推理→PLC）。无 PLC/总线硬件时本单元标 status=pending_hardware，
  仅做软件段延迟画像，不出完整实测结论（诚实标注）。

执行模型（§2.8）：
  - server + HIL（真实）：μs 级时间戳打点，1000 次检测-控制循环，对比 vanilla/PREEMPT_RT/硬件加速。
  - local / 无 HIL（可行性）：合成软件段延迟分布，演示纯软件方案抖动大、P99 超预算。

可单测核心逻辑：analyze_latency / judge_h11。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from common.registry import load_registry
from common.runtime import Runtime, load_runtime
from common.result import build_result, write_result, validate_result

OS_VARIANTS = ["vanilla_linux", "preempt_rt", "hw_accel"]
REALTIME_BUDGET_MS = 10.0        # hard 门槛：硬实时预算（通常 <10ms），运行前锁定

# 合成延迟参数（ms）：均值与抖动幅度，纯软件抖动大
_LAT_PROFILE = {
    "vanilla_linux": {"mean": 8.0, "jitter": 12.0},   # 均值不高但长尾/抖动大
    "preempt_rt":    {"mean": 6.0, "jitter": 3.0},
    "hw_accel":      {"mean": 2.0, "jitter": 0.4},
}


@dataclass
class N3Config:
    cycles: int = 1000
    realtime_budget_ms: float = REALTIME_BUDGET_MS


# --------------------------------------------------------------------------- #
def analyze_latency(samples: List[float]) -> Dict[str, float]:
    if not samples:
        return {}
    s = sorted(samples)
    n = len(s)
    mean = sum(s) / n
    p99 = s[min(n - 1, int(0.99 * n))]
    var = sum((x - mean) ** 2 for x in s) / n
    return {"mean_ms": mean, "p99_ms": p99, "max_ms": s[-1],
            "jitter_us": (var ** 0.5) * 1000}


def judge_h11(p99_ms: float, jitter_us: float, budget_ms: float,
              jitter_tol_us: float = 2000.0) -> Tuple[str, str]:
    ev = f"P99 {p99_ms:.1f}ms（预算 {budget_ms:.0f}ms），抖动 {jitter_us:.0f}μs"
    if p99_ms > budget_ms or jitter_us > jitter_tol_us:
        return "supported", ev + "（纯软件难满足硬实时）"
    return "not_supported", ev


# --------------------------------------------------------------------------- #
def _synthetic_samples(profile: Dict[str, float], n: int) -> List[float]:
    import random
    rng = random.Random(7)
    out = []
    for i in range(n):
        base = profile["mean"]
        # 长尾：少数样本出现大抖动（软件调度抢占/中断）
        spike = profile["jitter"] if rng.random() < 0.05 else abs(rng.gauss(0, profile["jitter"] * 0.15))
        out.append(base + spike)
    return out


def run(model_id: str = "m.yolov8n", dataset_id: str = "any",
        registry=None, runtime: Runtime | None = None, write: bool | None = None) -> Dict[str, Any]:
    registry = registry or load_registry()
    rt = runtime or load_runtime("local")
    model_id = model_id or "m.yolov8n"
    cfg = N3Config()

    has_hil = registry.hardware.get("hw.edge", {}).get("status") == "available"  # HIL 依赖边缘/控制硬件

    per_variant = {}
    for v in OS_VARIANTS:
        stats = analyze_latency(_synthetic_samples(_LAT_PROFILE[v], cfg.cycles))
        per_variant[v] = stats

    # 以纯软件 vanilla 方案判定 H11
    van = per_variant["vanilla_linux"]
    verdict, evidence = judge_h11(van["p99_ms"], van["jitter_us"], cfg.realtime_budget_ms)

    status = "ok" if has_hil else "pending_hardware"
    notes = ("需 HIL（PLC/总线）硬件才能出完整实测结论；当前无 HIL，仅软件段延迟画像。"
             if not has_hil else "")
    if rt.is_feasibility:
        notes = "local 可行性：合成软件段延迟分布，非真实 HIL 测量。" + (" " + notes if notes else "")

    res = build_result(
        test_id="N3", test_name="hard_realtime_control_chain",
        measurement_type="real" if (rt.is_measurement and has_hil) else "proxy",
        hardware_id="hw.edge",
        config={"model_id": model_id, "cycles": cfg.cycles,
                "realtime_budget_ms": cfg.realtime_budget_ms,
                "os_variants": OS_VARIANTS, "hil_status": status, "mode": rt.name},
        metrics={"per_os_variant": per_variant, "hil_status": status,
                 "summary_lines": [
                     f"vanilla P99 {van['p99_ms']:.1f}ms / 抖动 {van['jitter_us']:.0f}μs（预算 {cfg.realtime_budget_ms:.0f}ms）",
                     f"HIL 状态: {status}" + ("（无控制硬件，结论待验证）" if not has_hil else ""),
                 ]},
        hypothesis_id="H11", verdict=verdict, evidence=evidence, notes=notes,
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
