"""M2 —— 存储层次约束：量化-微缺陷抹除测试（Handbook §3 / 假设 H8）。

目的：验证低位宽量化（INT8/INT4）是否抹除微弱缺陷特征，尤其对依赖特征距离的异常检测方法。
判据：micro(<10px) 召回率下降幅度 显著大于 大缺陷 → H8 支持。

执行模型（§2.8）：
  - server（真实测量）：对模型导出各精度（TensorRT/ONNXRuntime/torch.ao），在含掩膜测试集上评估，
    按缺陷尺寸分层统计召回率。
  - local（可行性）：用合成样本 + 带"尺寸相关漏检"的合成评估器走通整条链路并演示判定逻辑，
    不写 result.json（真实测量仅在 server 产生）。

本模块中"与模型无关、可单测"的部分：recall_by_size / micro_recall_drop / verdict 判定，
以及精度矩阵编排。真实的量化推理由 adapter 在 server 上提供。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from common.registry import load_registry
from common.runtime import Runtime, load_runtime
from common.datasets import get_dataset, defect_size_bins, DEFAULT_SIZE_BINS
from common.models import get_adapter
from common.result import build_result, write_result, validate_result

# 精度矩阵（Handbook §3 M2）
PRECISIONS = ["fp32", "fp16", "int8_ptq", "int8_qat", "int4"]

# 名义压缩比/加速比（server 上由实际导出测得覆盖；此处为占位对照）
NOMINAL_SIZE_RATIO = {"fp32": 1.0, "fp16": 0.5, "int8_ptq": 0.25, "int8_qat": 0.25, "int4": 0.125}
NOMINAL_SPEEDUP = {"fp32": 1.0, "fp16": 1.5, "int8_ptq": 2.5, "int8_qat": 2.4, "int4": 3.0}

# 合成评估器：不同精度对不同尺寸缺陷的基础漏检率（仅 local 演示用）
_SYN_BASE_MISS = {"fp32": 0.0, "fp16": 0.02, "int8_ptq": 0.15, "int8_qat": 0.05, "int4": 0.35}
_SYN_SIZE_FACTOR = {"micro": 1.0, "small": 0.4, "large": 0.1}


@dataclass
class M2Config:
    model_id: str
    dataset_id: str
    precisions: List[str] = field(default_factory=lambda: list(PRECISIONS))
    calib_set_size: int = 200
    size_bins: List[Tuple[str, int, int]] = field(default_factory=lambda: list(DEFAULT_SIZE_BINS))
    n_eval: int = 120           # 评估样本数（local 合成时的规模，够填满三个尺寸箱）


# --------------------------------------------------------------------------- #
# 与模型无关、可单测的核心逻辑
# --------------------------------------------------------------------------- #
def defect_size_of(sample, bins=DEFAULT_SIZE_BINS) -> str | None:
    """取样本最大缺陷所属尺寸箱；正常样本返回 None。"""
    if sample.mask is None or isinstance(sample.mask, str):
        return None
    counts = defect_size_bins(sample.mask, bins)
    # 取存在缺陷的最大尺寸箱（large>small>micro 的逆序找第一个非零）
    for name, _, _ in reversed(bins):
        if counts.get(name, 0) > 0:
            return name
    return None


def recall_by_size(records: List[Tuple[str, bool]], bins=DEFAULT_SIZE_BINS) -> Dict[str, float]:
    """records: [(size_bin, detected)]；返回每个尺寸箱的召回率。"""
    out = {}
    for name, _, _ in bins:
        grp = [d for b, d in records if b == name]
        out[name] = (sum(1 for d in grp if d) / len(grp)) if grp else float("nan")
    return out


def micro_recall_drop(baseline: Dict[str, float], quant: Dict[str, float]) -> Dict[str, float]:
    """各尺寸箱：baseline 召回 − 量化召回（正=下降）。"""
    drop = {}
    for k in baseline:
        b, q = baseline.get(k), quant.get(k)
        drop[k] = (b - q) if (b == b and q == q) else float("nan")  # nan-safe
    return drop


def judge_h8(drops_by_precision: Dict[str, Dict[str, float]]) -> Tuple[str, str]:
    """H8 判定：最激进精度下 micro 下降 > large 下降 且 micro 下降>0 → supported。"""
    # 选“最激进”精度：int4 优先，否则 int8_ptq
    for prec in ["int4", "int8_ptq", "int8_qat", "fp16"]:
        if prec in drops_by_precision:
            d = drops_by_precision[prec]
            md, ld = d.get("micro", float("nan")), d.get("large", float("nan"))
            if md == md and ld == ld:
                if md > 0 and md > ld:
                    return "supported", f"{prec}: micro下降{md:.2f} > large下降{ld:.2f}"
                if md <= 0:
                    return "not_supported", f"{prec}: micro无下降({md:.2f})"
                return "partial", f"{prec}: micro下降{md:.2f} 未明显大于 large{ld:.2f}"
    return "unfilled", "无有效精度数据"


# --------------------------------------------------------------------------- #
# 真实量化评估：fp32(基线)/fp16(半精度)/int8_ptq(动态量化) 都是可以在通用 CPU/GPU 上
# 直接跑通的真实精度切换；int8_qat 需要量化感知重训、int4 需要专用推理库/硬件支持，
# 这两者超出了"拿到一个训练好的 checkpoint 就能测"的通用 adapter 范围，如实标注不支持，
# 不假装产出数字。
# --------------------------------------------------------------------------- #
_REAL_SUPPORTED_PRECISIONS = ("fp32", "fp16", "int8_ptq")


def _real_evaluate_precision(precision: str, samples, adapter, rt: Runtime, cfg: M2Config
                             ) -> List[Tuple[str, bool]]:
    if precision not in _REAL_SUPPORTED_PRECISIONS:
        raise NotImplementedError(
            f"M2 真实路径暂不支持精度 '{precision}'：int8_qat 需要量化感知重训、"
            f"int4 需要专用推理库（如 TensorRT/bitsandbytes），超出通用 checkpoint adapter 范围。"
        )

    torch = rt.optional_import("torch")
    raw_model = adapter.model.model   # 底层 nn.Module（PatchcoreModel/ReverseDistillationModel/...）
    original_dtype = next(raw_model.parameters()).dtype

    try:
        if precision == "fp32":
            active_model = raw_model
        elif precision == "fp16":
            raw_model.half()
            active_model = raw_model
        else:  # int8_ptq：动态量化（CPU 推理，权重转 int8，激活运行时量化）
            active_model = torch.ao.quantization.quantize_dynamic(
                raw_model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
            )

        scores, labels, size_bins = [], [], []
        for s in samples:
            arr = s.load_array()
            if arr is None:
                continue
            x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
            x = torch.nn.functional.interpolate(x, size=(adapter.input_hw, adapter.input_hw))
            if precision == "fp16":
                x = x.half()
            with torch.no_grad():
                out = active_model(x)
            score = getattr(out, "pred_score", out)
            score = float(score.flatten()[0]) if hasattr(score, "flatten") else float(score)
            is_anom = s.label == "anomaly"
            scores.append(score)
            labels.append(1 if is_anom else 0)
            size_bins.append(defect_size_of(s, cfg.size_bins) if is_anom else None)

        from compute.c6_upgrade_consistency import optimal_threshold
        thr = optimal_threshold(scores, labels)
        return [(sb, score >= thr) for score, sb in zip(scores, size_bins) if sb is not None]
    finally:
        raw_model.to(original_dtype)   # 精度切换是破坏性的（半精度/量化会替换权重），跑完必须还原


# --------------------------------------------------------------------------- #
# 评估器：真实（server）/ 合成（local）
# --------------------------------------------------------------------------- #
def _evaluate_precision(cfg: M2Config, precision: str, samples, adapter, rt: Runtime
                        ) -> List[Tuple[str, bool]]:
    """返回 [(size_bin, detected)]，仅统计有缺陷样本。"""
    records = []
    if rt.is_measurement and adapter.backend == "real":
        return _real_evaluate_precision(precision, samples, adapter, rt, cfg)
    else:
        # 合成评估器（local 可行性，确定性）：按 miss_prob 期望值对每个尺寸箱标注漏检，
        # 去除随机噪声，让 H8 模式（micro 掉最多）清晰可读。属带标注的逻辑演示，非真实测量。
        base = _SYN_BASE_MISS.get(precision, 0.1)
        by_bin: Dict[str, list] = {}
        for s in samples:
            sb = defect_size_of(s, cfg.size_bins)
            if sb is not None:
                by_bin.setdefault(sb, []).append(s)
        for sb, items in by_bin.items():
            miss = base * _SYN_SIZE_FACTOR.get(sb, 0.5)
            n_miss = round(miss * len(items))
            for i in range(len(items)):
                records.append((sb, i >= n_miss))   # 前 n_miss 个判为漏检
    return records


# --------------------------------------------------------------------------- #
# 编排入口
# --------------------------------------------------------------------------- #
def run(model_id: str | None = None, dataset_id: str | None = None,
        registry=None, runtime: Runtime | None = None,
        precisions: List[str] | None = None, write: bool | None = None) -> Dict[str, Any]:
    registry = registry or load_registry()
    rt = runtime or load_runtime("local")
    model_id = model_id or "m.patchcore"
    dataset_id = dataset_id or "d.mvtec"
    cfg = M2Config(model_id=model_id, dataset_id=dataset_id,
                   precisions=precisions or list(PRECISIONS))

    # 前置校验：M2 需要像素掩膜
    ds_spec = registry.datasets[dataset_id]
    if not ds_spec.get("has_mask"):
        raise ValueError(f"M2 需要含掩膜数据集，'{dataset_id}' 无掩膜")

    adapter = get_adapter(model_id, registry, rt).load()
    # 可行性下用较大合成集以覆盖 micro/small/large 三档；server 走真实测试集全量
    n_syn = cfg.n_eval if rt.is_feasibility else 8
    ds = get_dataset(dataset_id, registry, rt, split="test", n_synthetic=n_syn)
    samples = list(ds)

    per_precision: Dict[str, Any] = {}
    recalls: Dict[str, Dict[str, float]] = {}
    unsupported_precisions: List[str] = []
    for prec in cfg.precisions:
        try:
            recs = _evaluate_precision(cfg, prec, samples, adapter, rt)
        except NotImplementedError as e:
            unsupported_precisions.append(prec)
            per_precision[prec] = {"status": "unsupported", "reason": str(e)}
            continue
        rc = recall_by_size(recs, cfg.size_bins)
        recalls[prec] = rc
        per_precision[prec] = {
            "model_size_mb": None,             # server 由实际导出填
            "size_ratio": NOMINAL_SIZE_RATIO.get(prec),
            "speedup": NOMINAL_SPEEDUP.get(prec),
            "overall_metric": None,            # server 填 AUROC/mAP
            "recall_by_size": rc,
            "n_defect_samples": len(recs),
        }

    baseline = recalls.get("fp32", {})
    drops = {p: micro_recall_drop(baseline, recalls[p]) for p in recalls if p != "fp32"}
    for p in drops:
        per_precision[p]["recall_drop_by_size"] = drops[p]
    verdict, evidence = judge_h8(drops)

    measurement_type = "real" if rt.is_measurement else "proxy"
    res = build_result(
        test_id="M2", test_name="quantization_micro_defect_erasure",
        measurement_type=measurement_type, hardware_id="hw.cloud",
        config={"model_id": model_id, "dataset_id": dataset_id,
                "precisions": cfg.precisions, "calib_set_size": cfg.calib_set_size,
                "size_bins": [list(b) for b in cfg.size_bins],
                "mode": rt.name, "backend": adapter.backend},
        metrics={"per_precision": per_precision,
                 "micro_recall_drop_by_precision": drops,
                 "recall_by_size_by_precision": recalls},
        hypothesis_id="H8", verdict=verdict, evidence=evidence,
        notes=("local 可行性：合成评估器演示 H8 模式，非真实测量。" if rt.is_feasibility
              else ("server 真实：fp32/fp16/int8_ptq(动态量化) 为真实推理+按尺寸分层召回率。"
                    + (f" 不支持的精度（如实跳过，未产出数字）：{unsupported_precisions}。"
                       if unsupported_precisions else ""))),
    )

    issues = validate_result(res)
    if issues:
        raise RuntimeError("result 不合规: " + "; ".join(issues))

    do_write = rt.is_measurement if write is None else write
    if do_write:
        path = write_result(res)
        res["_written_to"] = str(path)
    return res


if __name__ == "__main__":
    import json, sys
    mid = sys.argv[1] if len(sys.argv) > 1 else "m.patchcore"
    did = sys.argv[2] if len(sys.argv) > 2 else "d.mvtec"
    r = run(mid, did)
    print(json.dumps({"verdict": r["hypothesis"], "recalls": r["metrics"]["recall_by_size_by_precision"]},
                     indent=2, ensure_ascii=False))
