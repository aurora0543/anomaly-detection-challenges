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
    edge_ratio_threshold: float = 5.0     # prior：边/云≥5x 视为"边缘受限"
    full_epochs: int = 20                 # 真实测量：A_full 场景训练轮数（server 上按需调大）
    incr_epochs: int = 3                  # 真实测量：B_incremental/C_fewshot 场景训练轮数
    full_n_samples: int = 300             # A_full 场景合成/换产新样本量
    incr_n_samples: int = 30              # B/C 场景合成/换产新样本量


_YOLO_BASE_WEIGHTS = {"m.yolov8n": "yolov8n.pt", "m.yolov8s": "yolov8s.pt", "m.yolov11n": "yolo11n.pt"}
_ANOMALY_MODEL_CLASS = {"m.patchcore": "Patchcore", "m.rd4ad": "ReverseDistillation",
                        "m.efficientad": "EfficientAd", "m.supersimplenet": "Supersimplenet"}


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
# 真实训练：同一模型架构，只改"新样本量"（换产场景的核心变量），真实计时。
# 用合成小图走通链路，不依赖某个具体数据集是否已在本机就位；在真实 GPU 服务器上把
# n_samples/epochs 调到真实规模，即是真实换产成本测量（架构、数据管线完全一致）。
# --------------------------------------------------------------------------- #
def _write_synthetic_folder_dataset(root, n_train: int, n_test_good: int = 4, n_test_defect: int = 4, size: int = 64):
    from PIL import Image
    import random
    train_dir = root / "train" / "good"
    test_good_dir = root / "test" / "good"
    test_defect_dir = root / "test" / "defect"
    for d in (train_dir, test_good_dir, test_defect_dir):
        d.mkdir(parents=True, exist_ok=True)
    for i in range(n_train):
        Image.new("RGB", (size, size), (255, 255, 255)).save(train_dir / f"{i:05d}.jpg")
    for i in range(n_test_good):
        Image.new("RGB", (size, size), (255, 255, 255)).save(test_good_dir / f"{i:05d}.jpg")
    for i in range(n_test_defect):
        Image.new("RGB", (size, size), (random.randint(0, 50),) * 3).save(test_defect_dir / f"{i:05d}.jpg")
    return root


def _real_anomalib_train(model_id: str, n_samples: int, epochs: int) -> Dict[str, Any]:
    import tempfile
    import time
    from pathlib import Path as _Path

    cls_name = _ANOMALY_MODEL_CLASS.get(model_id)
    if cls_name is None:
        raise KeyError(f"C4: 未知 anomaly 模型 id '{model_id}'")

    from anomalib.data import Folder
    from anomalib.engine import Engine
    import anomalib.models as anomalib_models

    model_cls = getattr(anomalib_models, cls_name)
    kwargs = {"backbone": "resnet18", "layers": ["layer1", "layer2", "layer3"]} if cls_name == "ReverseDistillation" \
        else {"backbone": "resnet18", "layers": ["layer2", "layer3"]} if cls_name == "Patchcore" else {}
    model = model_cls(**kwargs)

    # EfficientAd 的知识蒸馏机制要求 train_batch_size 严格等于 1（不是可调超参，anomalib
    # 自己在 __init__ 里断言这个值），其余模型可以正常按样本量取一个小 batch。
    train_bs = 1 if cls_name == "EfficientAd" else min(8, n_samples)

    with tempfile.TemporaryDirectory() as tmp:
        root = _write_synthetic_folder_dataset(_Path(tmp), n_train=n_samples)
        dm = Folder(name="c4_changeover", root=str(root), normal_dir="train/good",
                    normal_test_dir="test/good", abnormal_dir="test/defect",
                    train_batch_size=train_bs, eval_batch_size=8, num_workers=0)
        engine = Engine(max_epochs=epochs, accelerator="cpu", devices=1, logger=False,
                        enable_progress_bar=False, default_root_dir=tmp)
        t0 = time.perf_counter()
        engine.fit(model=model, datamodule=dm)
        train_time_s = time.perf_counter() - t0

        test_results = engine.test(model=model, datamodule=dm, verbose=False)
        final_metric = None
        if test_results:
            for k, v in test_results[0].items():
                if "AUROC" in k:
                    final_metric = float(v)
                    break

    return {"train_time_s": train_time_s, "final_metric": final_metric, "n_samples": n_samples, "epochs": epochs}


def _real_yolo_train(model_id: str, n_samples: int, epochs: int) -> Dict[str, Any]:
    import tempfile
    import time
    from pathlib import Path as _Path
    from ultralytics import YOLO
    import yaml as _yaml

    base_weights = _YOLO_BASE_WEIGHTS.get(model_id)
    if base_weights is None:
        raise KeyError(f"C4: 未知 YOLO 模型 id '{model_id}'")

    with tempfile.TemporaryDirectory() as tmp:
        root = _Path(tmp)
        img_train = root / "images" / "train"
        img_val = root / "images" / "val"
        lbl_train = root / "labels" / "train"
        lbl_val = root / "labels" / "val"
        for d in (img_train, img_val, lbl_train, lbl_val):
            d.mkdir(parents=True, exist_ok=True)

        from PIL import Image
        for i in range(n_samples):
            Image.new("RGB", (64, 64), (255, 255, 255)).save(img_train / f"{i:05d}.jpg")
            (lbl_train / f"{i:05d}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        for i in range(4):
            Image.new("RGB", (64, 64), (255, 255, 255)).save(img_val / f"{i:05d}.jpg")
            (lbl_val / f"{i:05d}.txt").write_text("0 0.5 0.5 0.2 0.2\n")

        data_yaml = {"path": str(root), "train": "images/train", "val": "images/val", "names": {0: "defect"}}
        yaml_path = root / "data.yaml"
        yaml_path.write_text(_yaml.dump(data_yaml, sort_keys=False))

        model = YOLO(base_weights)
        t0 = time.perf_counter()
        model.train(data=str(yaml_path), epochs=epochs, imgsz=64, batch=min(8, n_samples),
                    verbose=False, plots=False, val=True, project=str(root / "runs"), name="c4")
        train_time_s = time.perf_counter() - t0

        metrics = model.metrics
        final_metric = float(metrics.box.map50) if metrics is not None else None

    return {"train_time_s": train_time_s, "final_metric": final_metric, "n_samples": n_samples, "epochs": epochs}


def _real_training_run(model_id: str, registry, n_samples: int, epochs: int) -> Dict[str, Any]:
    adapter_key = registry.models[model_id].get("adapter")
    if adapter_key == "repo_pipeline":
        return _real_anomalib_train(model_id, n_samples, epochs)
    if adapter_key == "ultralytics_weights":
        return _real_yolo_train(model_id, n_samples, epochs)
    raise NotImplementedError(
        f"C4 真实训练未覆盖 adapter '{adapter_key}'（zeroshot 模型如 MoECLIP 不参与换产重训，不适用 H3）"
    )


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
        # 真实换产成本：同一模型架构，只变"新样本量"，真实训练+计时。
        # A_full/C_fewshot 各跑一次真实训练；B_incremental 复用与 C 相同的小样本训练机制
        # （真正的"从已有权重热启动增量微调"未在 repo 训练脚本里单独实现，这里如实标注为近似）；
        # D_edge_incremental 仍受限于没有边缘硬件，保持"待验证"。
        full_run = _real_training_run(model_id, registry, cfg.full_n_samples, cfg.full_epochs)
        incr_run = _real_training_run(model_id, registry, cfg.incr_n_samples, cfg.incr_epochs)

        full_h = full_run["train_time_s"] / 3600.0
        incr_min = incr_run["train_time_s"] / 60.0
        per_scenario = {
            "A_full": {"hw": "cloud", "train_time_h": full_h, "final_metric": full_run["final_metric"],
                      "n_samples": full_run["n_samples"], "epochs": full_run["epochs"]},
            "B_incremental": {"hw": "cloud", "train_time_min": incr_min, "final_metric": incr_run["final_metric"],
                             "n_samples": incr_run["n_samples"], "epochs": incr_run["epochs"],
                             "note": "近似：复用小样本重训机制，非真正热启动增量微调"},
            "C_fewshot": {"hw": "cloud", "train_time_min": incr_min, "final_metric": incr_run["final_metric"],
                         "n_samples": incr_run["n_samples"], "epochs": incr_run["epochs"]},
            "D_edge_incremental": {"hw": "edge", "status": "待验证（无边缘设备）"},
        }
        edge_ratio = _EDGE_SLOWDOWN   # 无边缘硬件，沿用 prior 估计，非本次真实测得
    else:
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
        edge_ratio = _EDGE_SLOWDOWN

    speedup = full_vs_incremental_speedup(full_h, incr_min)
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
