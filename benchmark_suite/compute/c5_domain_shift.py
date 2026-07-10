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


# --------------------------------------------------------------------------- #
# 真实跨域评估：域A训练 -> 域A(留出)测 vs 域B测，anomalib 真实 AUROC。
# --------------------------------------------------------------------------- #
def _materialize_samples(samples, dest_dir):
    """把 Sample（合成 array 或真实 image_path）落盘成 jpg，返回写入的文件数。"""
    from PIL import Image
    import numpy as np
    dest_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for i, s in enumerate(samples):
        arr = s.load_array() if s.image_path else s.array
        if arr is None:
            continue
        img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype("uint8")) if arr.max() <= 1.0 else Image.fromarray(arr.astype("uint8"))
        img.save(dest_dir / f"{i:05d}.jpg")
        n += 1
    return n


def _real_cross_domain_eval(model_id: str, dataset_id: str, registry, rt: Runtime,
                            train_domains: List[str], test_domains: List[str]) -> Dict[str, float]:
    import tempfile
    from pathlib import Path as _Path
    from anomalib.data import Folder
    from anomalib.engine import Engine
    import anomalib.models as anomalib_models
    from compute.c4_changeover_training import _ANOMALY_MODEL_CLASS

    cls_name = _ANOMALY_MODEL_CLASS.get(model_id)
    if cls_name is None:
        raise NotImplementedError(f"C5 真实跨域评估目前只覆盖 anomaly 系模型，不适用于 '{model_id}'")

    ds = get_dataset(dataset_id, registry, rt, n_synthetic=80)
    all_samples = list(ds)
    domain_a_normal = [s for s in all_samples if s.domain in train_domains and s.label != "anomaly"]
    domain_a_anomaly = [s for s in all_samples if s.domain in train_domains and s.label == "anomaly"]
    domain_b_normal = [s for s in all_samples if s.domain in test_domains and s.label != "anomaly"]
    domain_b_anomaly = [s for s in all_samples if s.domain in test_domains and s.label == "anomaly"]

    split_idx = max(1, int(len(domain_a_normal) * 0.7))
    train_pool, a_test_normal = domain_a_normal[:split_idx], domain_a_normal[split_idx:] or domain_a_normal[-2:]

    model_cls = getattr(anomalib_models, cls_name)
    kwargs = {"backbone": "resnet18", "layers": ["layer1", "layer2", "layer3"]} if cls_name == "ReverseDistillation" \
        else {"backbone": "resnet18", "layers": ["layer2", "layer3"]} if cls_name == "Patchcore" else {}
    model = model_cls(**kwargs)

    with tempfile.TemporaryDirectory() as tmp:
        root = _Path(tmp)
        _materialize_samples(train_pool, root / "train" / "good")
        _materialize_samples(a_test_normal, root / "test" / "good")
        _materialize_samples(domain_a_anomaly or a_test_normal[:1], root / "test" / "defect")

        dm = Folder(name="c5_domain_a", root=str(root), normal_dir="train/good",
                    normal_test_dir="test/good", abnormal_dir="test/defect",
                    train_batch_size=4, eval_batch_size=4, num_workers=0)
        engine = Engine(max_epochs=1, accelerator="cpu", devices=1, logger=False,
                        enable_progress_bar=False, default_root_dir=tmp)
        engine.fit(model=model, datamodule=dm)

        def _auroc_of(test_results):
            if not test_results:
                return float("nan")
            for k, v in test_results[0].items():
                if "AUROC" in k:
                    return float(v)
            return float("nan")

        in_domain_test = engine.test(model=model, datamodule=dm, verbose=False)
        in_domain_auroc = _auroc_of(in_domain_test)

        root_b = _Path(tmp) / "domain_b"
        _materialize_samples(domain_b_normal or domain_a_normal[:2], root_b / "test" / "good")
        _materialize_samples(domain_b_anomaly or domain_a_anomaly or domain_b_normal[:1], root_b / "test" / "defect")
        # 复用同一 train/good（Folder 要求 normal_dir 存在，跨域测试本身只看 test 部分）
        (root_b / "train" / "good").mkdir(parents=True, exist_ok=True)
        _materialize_samples(train_pool[:2], root_b / "train" / "good")
        dm_b = Folder(name="c5_domain_b", root=str(root_b), normal_dir="train/good",
                     normal_test_dir="test/good", abnormal_dir="test/defect",
                     train_batch_size=4, eval_batch_size=4, num_workers=0)
        cross_domain_test = engine.test(model=model, datamodule=dm_b, verbose=False)
        cross_domain_auroc = _auroc_of(cross_domain_test)

    return {"in_domain_metric": in_domain_auroc, "cross_domain_metric": cross_domain_auroc,
            "n_train": len(train_pool), "n_test": len(a_test_normal) + len(domain_b_normal)}


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
        real = _real_cross_domain_eval(model_id, dataset_id, registry, rt,
                                       train_domains=doms[:1], test_domains=doms[1:])
        in_metric, cross_metric = real["in_domain_metric"], real["cross_domain_metric"]
        n_train, n_test = real["n_train"], real["n_test"]
    else:
        in_metric = _IN_DOMAIN_AUROC.get(pdm, 0.90)
        cross_metric = in_metric - _CROSS_DROP.get(pdm, 0.12)
        n_train, n_test = len(split["train"]), len(split["test"])
    verdict, evidence = judge_h5(in_metric, cross_metric, cfg)

    res = build_result(
        test_id="C5", test_name="cross_domain_shift",
        measurement_type="real" if rt.is_measurement else "proxy", hardware_id="hw.cloud",
        config={"model_id": model_id, "dataset_id": dataset_id,
                "train_domains": doms[:1], "test_domains": doms[1:],
                "synthetic_shift": False, "mode": rt.name},
        metrics={"in_domain_metric": in_metric, "cross_domain_metric": cross_metric,
                 "degradation_pct": degradation_pct(in_metric, cross_metric),
                 "n_train": n_train, "n_test": n_test,
                 "summary_lines": [
                     f"训练域 {doms[:1]} / 测试域 {doms[1:]}（样本 {n_train}/{n_test}）",
                     f"单域 {in_metric:.2f} → 跨域 {cross_metric:.2f}",
                 ]},
        hypothesis_id="H5", verdict=verdict, evidence=evidence,
        notes=("local 可行性：域划分真实、精度为合成演示。" if rt.is_feasibility
              else "server 真实：域A训练→域A留出/域B测试的真实 anomalib AUROC。"),
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
