"""C5 —— 数据分布异构性：跨域泛化与漂移测试（Handbook §3 / 假设 H5）。

目的：量化跨工位（跨域）性能退化，检验中心统一模型的漂移代价。
判据（prior）：跨域精度 显著低于 单域精度（degradation_pct > 0）→ H5 支持。

关键：用天然多源域划分（ZJU 纹理型 / VisA 类目），禁止用光照/噪声增广冒充真实漂移。

执行模型（§2.8）：
  - server（真实）：域A训→域A/域B测（异常检测模型）；可选单轮 FedAvg 探针。
  - local（可行性）：用 split_by_domain 真实切分样本，合成单域/跨域精度演示退化。

可单测核心逻辑：degradation_pct / judge_h5。
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from common.registry import load_registry
from common.runtime import Runtime, load_runtime
from common.models import get_adapter
from common.datasets import get_dataset, split_by_domain
from common.result import build_result, write_result, validate_result

# 合成精度（local 演示）：单域高、跨域退化
_IN_DOMAIN_AUROC = {"feature_embed": 0.94, "conv": 0.90, "attention": 0.91}
_CROSS_DROP = {"feature_embed": 0.18, "conv": 0.12, "attention": 0.10}   # 绝对退化


@dataclass
class C5Config:
    min_degradation: float = 0.02        # prior：退化>2% 视为显著


def degradation_pct(in_metric: float, cross_metric: float) -> float:
    return (in_metric - cross_metric) / in_metric if in_metric else float("nan")


def judge_h5(in_metric: float, cross_metric: float, cfg: C5Config) -> Tuple[str, str]:
    deg = degradation_pct(in_metric, cross_metric)
    ev = f"单域 {in_metric:.2f} → 跨域 {cross_metric:.2f}（退化 {deg*100:.0f}%）"
    if deg != deg:
        return "unfilled", "无精度数据"
    if deg > cfg.min_degradation:
        return "supported", ev
    return "not_supported", ev


def run(model_id: str = "m.patchcore", dataset_id: str = "d.zju",
        registry=None, runtime: Runtime | None = None, write: bool | None = None) -> Dict[str, Any]:
    registry = registry or load_registry()
    rt = runtime or load_runtime("local")
    model_id = model_id or "m.patchcore"
    dataset_id = dataset_id or "d.zju"
    cfg = C5Config()
    adapter = get_adapter(model_id, registry, rt).load()
    pdm = registry.models[model_id].get("paradigm")

    # 用真实/合成域划分（证明接口贯通）
    ds = get_dataset(dataset_id, registry, rt, n_synthetic=60)
    doms = ds.domains()
    if len(doms) < 2:
        raise ValueError(f"C5 需 ≥2 个域，'{dataset_id}' 域字段不足")
    split = split_by_domain(ds, train_domains=doms[:1], test_domains=doms[1:])

    if rt.is_measurement and adapter.backend == "real":
        raise NotImplementedError("C5 真实跨域评估在 server 步骤：域A训练→域B测试")

    in_metric = _IN_DOMAIN_AUROC.get(pdm, 0.90)
    cross_metric = in_metric - _CROSS_DROP.get(pdm, 0.12)
    verdict, evidence = judge_h5(in_metric, cross_metric, cfg)

    res = build_result(
        test_id="C5", test_name="cross_domain_shift",
        measurement_type="real" if rt.is_measurement else "proxy", hardware_id="hw.cloud",
        config={"model_id": model_id, "dataset_id": dataset_id,
                "train_domains": doms[:1], "test_domains": doms[1:],
                "synthetic_shift": False, "mode": rt.name},
        metrics={"in_domain_metric": in_metric, "cross_domain_metric": cross_metric,
                 "degradation_pct": degradation_pct(in_metric, cross_metric),
                 "n_train": len(split["train"]), "n_test": len(split["test"]),
                 "summary_lines": [
                     f"训练域 {doms[:1]} / 测试域 {doms[1:]}（样本 {len(split['train'])}/{len(split['test'])}）",
                     f"单域 {in_metric:.2f} → 跨域 {cross_metric:.2f}",
                 ]},
        hypothesis_id="H5", verdict=verdict, evidence=evidence,
        notes=("local 可行性：域划分真实、精度为合成演示。" if rt.is_feasibility else ""),
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
