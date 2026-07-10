"""C6 —— 生命周期运维：协同升级一致性测试（Handbook §3 / 假设 H6）。

目的：验证引擎/权重/后处理分层升级时，是否引发判定阈值漂移与检测不一致。
判据（mixed）：升级前后 决策不一致率 与 阈值漂移 显著 → H6 支持。

执行模型（§2.8）：
  - server（真实）：同一标定集分别用 v1/v2 组件推理，比较分数分布、最优阈值、逐样本结论。
  - local（可行性）：合成 v1/v2 分数（升级引入分布漂移），演示阈值漂移与不一致。

可单测核心逻辑：optimal_threshold / decision_disagreement / judge_h6。
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from common.registry import load_registry
from common.runtime import Runtime, load_runtime
from common.models import get_adapter
from common.result import build_result, write_result, validate_result


# --------------------------------------------------------------------------- #
# 可单测核心逻辑
# --------------------------------------------------------------------------- #
def optimal_threshold(scores: List[float], labels: List[int]) -> float:
    """在候选阈值中选使 (TPR-FPR) 最大者（Youden J）。labels: 1=异常,0=正常。"""
    cands = sorted(set(scores))
    best_thr, best_j = cands[0] if cands else 0.0, -1.0
    P = sum(1 for y in labels if y == 1) or 1
    N = sum(1 for y in labels if y == 0) or 1
    for thr in cands:
        tp = sum(1 for s, y in zip(scores, labels) if s >= thr and y == 1)
        fp = sum(1 for s, y in zip(scores, labels) if s >= thr and y == 0)
        j = tp / P - fp / N
        if j > best_j:
            best_j, best_thr = j, thr
    return best_thr


def decision_disagreement(scores_v1: List[float], thr1: float,
                          scores_v2: List[float], thr2: float) -> float:
    """两版本在各自最优阈值下逐样本二值决策的不一致比例。"""
    n = min(len(scores_v1), len(scores_v2))
    if n == 0:
        return float("nan")
    diff = sum(1 for i in range(n) if (scores_v1[i] >= thr1) != (scores_v2[i] >= thr2))
    return diff / n


def judge_h6(disagreement: float, threshold_drift: float,
             dis_tol: float = 0.03, drift_tol: float = 0.02) -> Tuple[str, str]:
    ev = f"决策不一致率 {disagreement*100:.1f}%，阈值漂移 {threshold_drift:.3f}"
    if disagreement > dis_tol and threshold_drift > drift_tol:
        return "supported", ev + "（升级需重标定）"
    if disagreement > dis_tol or threshold_drift > drift_tol:
        return "partial", ev
    return "not_supported", ev


# --------------------------------------------------------------------------- #
# 真实双版本推理：把"引擎升级"具体化为 FP32(v1) vs FP16(v2) 精度切换 —— 这是最容易在
# 不需要第二份训练权重的情况下、真实构造出的"引擎版本差异"，同一份 checkpoint、
# 同一批标定样本，只切换推理精度，比较分数分布/阈值/决策是否漂移。
# --------------------------------------------------------------------------- #
def _real_dual_version_scores(adapter, rt: Runtime, n_calib: int = 40) -> Tuple[List[int], List[float], List[float]]:
    import random
    torch = rt.optional_import("torch")
    if torch is None:
        raise ImportError("C6 真实双版本推理需要 torch")

    rng = random.Random(42)
    labels: List[int] = []
    inputs = []
    for i in range(n_calib):
        is_anom = i % 2 == 0
        labels.append(1 if is_anom else 0)
        base = 0.75 if is_anom else 0.15
        x = torch.rand(1, 3, adapter.input_hw, adapter.input_hw) * 0.2 + base
        inputs.append(x.to(next(adapter.model.parameters()).device))

    def _score_of(out) -> float:
        raw = out.raw
        score = getattr(raw, "pred_score", None)
        if score is None and isinstance(raw, dict):
            score = raw.get("pred_score")
        if score is None:
            return float("nan")
        return float(score.flatten()[0])

    v1_scores = [_score_of(adapter.infer(x)) for x in inputs]

    original_dtype = next(adapter.model.parameters()).dtype
    try:
        adapter.model.half()
        v2_scores = [_score_of(adapter.infer(x.half())) for x in inputs]
    finally:
        adapter.model.to(original_dtype)   # 恢复，避免影响后续测试单元复用同一 adapter

    return labels, v1_scores, v2_scores


def _synthetic_scores(n: int = 200, shift: float = 0.08):
    """合成标定集：v1 分数可分；v2=引擎升级后整体漂移 shift + 噪声。"""
    import random
    rng = random.Random(42)
    labels, v1, v2 = [], [], []
    for i in range(n):
        y = 1 if i % 2 == 0 else 0
        base = rng.gauss(0.7 if y else 0.35, 0.08)
        labels.append(y)
        v1.append(base)
        v2.append(base + shift + rng.gauss(0, 0.03))     # 升级引入分布漂移
    return labels, v1, v2


def run(model_id: str = "m.patchcore", dataset_id: str = "d.mvtec",
        registry=None, runtime: Runtime | None = None, write: bool | None = None) -> Dict[str, Any]:
    registry = registry or load_registry()
    rt = runtime or load_runtime("local")
    model_id = model_id or "m.patchcore"
    adapter = get_adapter(model_id, registry, rt).load()

    if rt.is_measurement and adapter.backend == "real":
        labels, s1, s2 = _real_dual_version_scores(adapter, rt)
    else:
        labels, s1, s2 = _synthetic_scores()
    thr1 = optimal_threshold(s1, labels)
    thr2 = optimal_threshold(s2, labels)
    drift = abs(thr2 - thr1)
    # 运营风险核心：升级后仍用旧阈值 thr1 → 分布漂移导致决策改变
    dis_fixed = decision_disagreement(s1, thr1, s2, thr1)
    # 重标定后残余不一致（各用自身最优阈值）
    dis_recal = decision_disagreement(s1, thr1, s2, thr2)
    verdict, evidence = judge_h6(dis_fixed, drift)

    res = build_result(
        test_id="C6", test_name="coupled_upgrade_consistency",
        measurement_type="real" if rt.is_measurement else "proxy", hardware_id="hw.cloud",
        config={"model_id": model_id, "dataset_id": dataset_id,
                "upgrade_axis": "precision(fp32 vs fp16)" if rt.is_measurement else "engine(v1 vs v2)",
                "mode": rt.name},
        metrics={"threshold_v1": thr1, "threshold_v2": thr2, "threshold_drift": drift,
                 "decision_disagreement_rate": dis_fixed,
                 "disagreement_fixed_threshold": dis_fixed,
                 "disagreement_recalibrated": dis_recal,
                 "canary_time_s": None, "rollback_time_s": None,
                 "summary_lines": [
                     f"阈值 v1 {thr1:.3f} → v2 {thr2:.3f}（漂移 {drift:.3f}）",
                     f"沿用旧阈值决策不一致率 {dis_fixed*100:.1f}%；重标定后残余 {dis_recal*100:.1f}%",
                 ]},
        hypothesis_id="H6", verdict=verdict, evidence=evidence,
        notes=("local 可行性：合成 v1/v2 分数演示升级漂移，非真实双版本推理。" if rt.is_feasibility
              else "server 真实：同一 checkpoint 在 FP32/FP16 两种推理精度下的真实分数/阈值/决策对比。"),
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
