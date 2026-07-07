"""兼容性解析器 —— 把「模型↔数据集↔测试」的合法性规则写成代码（Handbook §2.4 / 评估 P7）。

三层规则：
  (A) 模型 ↔ 数据集：模型的训练范式决定它需要什么标注/训练集。
      - detection（有监督）  需框标注：dataset.format ∈ {yolo, coco}
      - anomaly（无监督）    需仅正常样本训练集：dataset.train_normal_only == True
      - zeroshot_anomaly     不训练，任何数据集测试集皆可（仅评估）
  (B) 测试 ↔ 数据集：测试的 needs 决定对数据集的要求。
      - pixel_mask（如 M2 尺寸分层）        需 has_mask == True
      - natural_domain_split（如 C5 跨域）  需 domain_field 非空
  (C) 测试 ↔ 模型：
      - train_to_convergence（如 C4）       需可训练模型（非 zeroshot）

对外接口：
  model_dataset_compatible / test_dataset_compatible / test_model_compatible
  check(registry, test_id, model_id, dataset_id) -> (ok, reasons)
  legal_combos(registry, test_id) -> List[{model, dataset}]
"""
from __future__ import annotations
from typing import Dict, List, Tuple

_SPECIAL = {"all", "any"}


# --------------------------------------------------------------------------- #
# 展开测试单元声明的模型/数据集集合（all/any → 全部）
# --------------------------------------------------------------------------- #
def expand_models(registry, test: dict) -> List[str]:
    mval = test.get("models")
    if isinstance(mval, str):
        return list(registry.models) if mval in _SPECIAL else [mval]
    return [m for m in (mval or []) if m in registry.models] or list(registry.models)


def expand_datasets(registry, test: dict) -> List[str]:
    dsv = test.get("datasets") or []
    if isinstance(dsv, str):
        dsv = [dsv]
    if any(d in _SPECIAL for d in dsv):
        return list(registry.datasets)
    return [d for d in dsv if d in registry.datasets]


# --------------------------------------------------------------------------- #
# (A) 模型 ↔ 数据集
# --------------------------------------------------------------------------- #
def model_dataset_compatible(model_spec: dict, ds_spec: dict) -> Tuple[bool, str]:
    task = model_spec.get("task")
    fmt = ds_spec.get("format")
    if task == "detection":
        if fmt in ("yolo", "coco"):
            return True, ""
        return False, f"有监督检测需框标注，数据集为 {fmt}（无框）"
    if task == "anomaly":
        if ds_spec.get("train_normal_only"):
            return True, ""
        if ds_spec.get("normal_subset_available"):
            return True, "由正常子集训练（数据集含异常但正常图充足）"
        return False, "无监督异常检测需仅正常样本训练集，该数据集训练集含异常且无正常子集"
    if task == "zeroshot_anomaly":
        return True, "零样本：仅用测试集评估"
    return True, ""


# --------------------------------------------------------------------------- #
# (B) 测试 ↔ 数据集
# --------------------------------------------------------------------------- #
def test_dataset_compatible(test_spec: dict, ds_spec: dict) -> Tuple[bool, str]:
    needs = test_spec.get("needs", []) or []
    if "pixel_mask" in needs and not ds_spec.get("has_mask"):
        return False, "测试需像素掩膜(尺寸分层)，数据集无 mask"
    if "natural_domain_split" in needs and not ds_spec.get("domain_field"):
        return False, "测试需天然域划分，数据集无 domain 字段"
    return True, ""


# --------------------------------------------------------------------------- #
# (C) 测试 ↔ 模型
# --------------------------------------------------------------------------- #
def test_model_compatible(test_spec: dict, model_spec: dict) -> Tuple[bool, str]:
    needs = test_spec.get("needs", []) or []
    if "train_to_convergence" in needs and model_spec.get("task") == "zeroshot_anomaly":
        return False, "测试需可训练模型，零样本模型不训练"
    return True, ""


# --------------------------------------------------------------------------- #
# 综合判定与枚举
# --------------------------------------------------------------------------- #
def check(registry, test_id: str, model_id: str, dataset_id: str) -> Tuple[bool, List[str]]:
    test = registry.test_units.get(test_id, {})
    m = registry.models.get(model_id)
    d = registry.datasets.get(dataset_id)
    reasons: List[str] = []
    if m is None:
        reasons.append(f"未知模型 {model_id}")
    if d is None:
        reasons.append(f"未知数据集 {dataset_id}")
    if reasons:
        return False, reasons
    for ok, why in (model_dataset_compatible(m, d),
                    test_dataset_compatible(test, d),
                    test_model_compatible(test, m)):
        if not ok:
            reasons.append(why)
    return (len(reasons) == 0), reasons


def legal_combos(registry, test_id: str) -> List[Dict[str, str]]:
    test = registry.test_units.get(test_id, {})
    combos = []
    for mid in expand_models(registry, test):
        for did in expand_datasets(registry, test):
            ok, _ = check(registry, test_id, mid, did)
            if ok:
                combos.append({"model": mid, "dataset": did})
    return combos


def all_legal(registry) -> Dict[str, List[Dict[str, str]]]:
    return {tid: legal_combos(registry, tid) for tid in registry.test_units}
