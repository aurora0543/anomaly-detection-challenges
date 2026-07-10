"""Cross-model comparison: inference latency/throughput vs. parameter count vs. accuracy.

Scope (per user request): compare models against each other only, all on the same
ZJU-Leaper pattern1 dataset. No external/absolute benchmarks.

Covers:
  - Patchcore, ReverseDistillation (RD4AD), EfficientAd, Supersimplenet
    (models/anomaly_detection_for_textile_industry/checkpoints/*.ckpt)
  - YOLOv8n, YOLOv8s, YOLOv11n (models/fabric-defect-detection/weights/*.pt)
Excluded per user: MoECLIP (not trained yet).

Accuracy numbers are NOT re-measured here - they're parsed straight from the
evaluation reports/metrics.json each training run already produced (all on the
same held-out test set), since re-running "the same dataset" evaluation would
just reproduce the same numbers. Latency/throughput/param-count ARE freshly,
really measured on this machine (Apple Silicon GPU via MPS, not CPU) so all
7 models are timed on identical hardware (the only way a cross-model
comparison is fair). No cloud CUDA timing exists to compare against - the
cloud training runs never persisted per-image speed logs (checked: ultralytics
results.csv has no speed column, and the anomalib Engine calls used
logger=False) - so this is MPS-only, not "cloud vs local".

Usage:
    python analysis/compare_models.py
Writes analysis/results/comparison.json and prints a summary table.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
ANOMALY_ROOT = REPO_ROOT / "models" / "anomaly_detection_for_textile_industry"
FABRIC_ROOT = REPO_ROOT / "models" / "fabric-defect-detection"
OUT_DIR = Path(__file__).resolve().parent / "results"

ANOMALY_INPUT_HW = 256   # general_configuration.image_size in config.yaml at train time
YOLO_IMGSZ = 512         # imgsz used by fabric-defect-detection/train.py
WARMUP_ITERS = 3
TIMED_ITERS = 20


def _pick_device() -> str:
    """Apple Silicon GPU (MPS) if available, else CUDA, else CPU. This Mac has an
    M-series GPU reachable via MPS - not CPU-only as an earlier pass wrongly assumed."""
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _sync(device: str):
    import torch
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


@dataclass
class ModelStats:
    name: str
    task: str
    total_params_m: float
    trainable_params_m: float
    latency_ms_mean: float
    latency_ms_p95: float
    fps: float
    accuracy: dict[str, Any]
    source: str
    device: str


# --------------------------------------------------------------------------- #
# Accuracy extraction: parse straight from the reports these training runs
# already produced, rather than re-measuring the same dataset again.
# --------------------------------------------------------------------------- #
def _latest_report_per_model() -> dict[str, Path]:
    """Map model name -> its most recent evaluation_report .txt, using the
    timestamp-matched config snapshot (whose filename embeds the model name)
    to identify which report belongs to which model."""
    config_dir = ANOMALY_ROOT / "results" / "results" / "EfficientAd" / "config"
    report_dir = ANOMALY_ROOT / "results" / "results" / "EfficientAd" / "report"
    ts_to_model: dict[str, str] = {}
    for cfg_file in config_dir.glob("*.yaml"):
        m = re.match(r"(\d{8}_\d{6})_(\w+)_config_", cfg_file.name)
        if m:
            ts_to_model[m.group(1)] = m.group(2)

    latest: dict[str, tuple[str, Path]] = {}
    for report_file in report_dir.glob("*_evaluation_report_*.txt"):
        m = re.match(r"(\d{8}_\d{6})_evaluation_report_", report_file.name)
        if not m:
            continue
        ts = m.group(1)
        model = ts_to_model.get(ts)
        if model is None:
            continue
        if model not in latest or ts > latest[model][0]:
            latest[model] = (ts, report_file)
    return {model: path for model, (ts, path) in latest.items()}


def _parse_anomaly_report(path: Path) -> dict[str, Any]:
    text = path.read_text()
    fields = ("Accuracy", "Precision", "Recall", "F1-Score", "AUROC", "AUPRO", "AP-loc")
    out = {}
    for f in fields:
        m = re.search(rf"{re.escape(f)}\s*:\s*([\d.]+)%", text)
        if m:
            out[f.lower().replace("-", "_")] = float(m.group(1))
    return out


def _parse_yolo_metrics(path: Path) -> dict[str, Any]:
    d = json.loads(path.read_text())
    return {"precision": d["precision"] * 100, "recall": d["recall"] * 100,
            "map50": d["map50"] * 100, "map50_95": d["map50-95"] * 100}


# --------------------------------------------------------------------------- #
# Real timing helper
# --------------------------------------------------------------------------- #
def _time_calls(fn, warmup: int, iters: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    lat = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        lat.append((time.perf_counter() - t0) * 1000.0)
    lat.sort()
    mean = sum(lat) / len(lat)
    p95 = lat[int(len(lat) * 0.95) - 1]
    return mean, p95


# --------------------------------------------------------------------------- #
def _measure_anomaly_model(model_key: str, ckpt_path: Path, accuracy: dict, device: str) -> ModelStats:
    import torch
    import anomalib.models as anomalib_models

    cls_name = {"Patchcore": "Patchcore", "ReverseDistillation": "ReverseDistillation",
               "EfficientAd": "EfficientAd", "Supersimplenet": "Supersimplenet"}[model_key]
    model_cls = getattr(anomalib_models, cls_name)
    lm = model_cls.load_from_checkpoint(str(ckpt_path), map_location=device, weights_only=False)
    lm.eval()
    lm.to(device)

    raw = lm.model
    total = sum(p.numel() for p in raw.parameters()) / 1e6
    trainable = sum(p.numel() for p in raw.parameters() if p.requires_grad) / 1e6

    x = torch.rand(1, 3, ANOMALY_INPUT_HW, ANOMALY_INPUT_HW, device=device)

    def _call():
        raw(x)
        _sync(device)

    with torch.no_grad():
        mean_ms, p95_ms = _time_calls(_call, WARMUP_ITERS, TIMED_ITERS)

    return ModelStats(
        name=model_key, task="anomaly_detection", total_params_m=total, trainable_params_m=trainable,
        latency_ms_mean=mean_ms, latency_ms_p95=p95_ms, fps=1000.0 / mean_ms if mean_ms else float("nan"),
        accuracy=accuracy, source=str(ckpt_path.relative_to(REPO_ROOT)), device=device,
    )


def _measure_yolo_model(model_key: str, weights_path: Path, accuracy: dict, device: str) -> ModelStats:
    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    model.to(device)
    raw = model.model
    total = sum(p.numel() for p in raw.parameters()) / 1e6
    trainable = sum(p.numel() for p in raw.parameters() if p.requires_grad) / 1e6

    import numpy as np
    dummy = (np.random.rand(YOLO_IMGSZ, YOLO_IMGSZ, 3) * 255).astype("uint8")

    def _call():
        model.predict(source=dummy, imgsz=YOLO_IMGSZ, verbose=False, device=device)
        _sync(device)

    mean_ms, p95_ms = _time_calls(_call, WARMUP_ITERS, TIMED_ITERS)

    return ModelStats(
        name=model_key, task="object_detection", total_params_m=total, trainable_params_m=trainable,
        latency_ms_mean=mean_ms, latency_ms_p95=p95_ms, fps=1000.0 / mean_ms if mean_ms else float("nan"),
        accuracy=accuracy, source=str(weights_path.relative_to(REPO_ROOT)), device=device,
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = _pick_device()
    print(f"Using device: {device}")
    reports = _latest_report_per_model()
    print(f"Found reports for: {sorted(reports)}")

    checkpoint_map = {
        "Patchcore": ANOMALY_ROOT / "checkpoints" / "last.ckpt",
        "ReverseDistillation": ANOMALY_ROOT / "checkpoints" / "ReverseDistillation-latest.ckpt",
        "EfficientAd": ANOMALY_ROOT / "checkpoints" / "EfficientAd-latest.ckpt",
        "Supersimplenet": ANOMALY_ROOT / "checkpoints" / "last-v2.ckpt",
    }

    results: list[ModelStats] = []
    for model_key, ckpt_path in checkpoint_map.items():
        if not ckpt_path.exists():
            print(f"[SKIP] {model_key}: checkpoint not found at {ckpt_path}")
            continue
        if model_key not in reports:
            print(f"[SKIP] {model_key}: no evaluation report found")
            continue
        print(f"Measuring {model_key} ...")
        accuracy = _parse_anomaly_report(reports[model_key])
        stats = _measure_anomaly_model(model_key, ckpt_path, accuracy, device)
        results.append(stats)
        print(f"  {stats.total_params_m:.1f}M params, {stats.latency_ms_mean:.1f}ms/{stats.fps:.1f}fps")

    yolo_weights = {
        "YOLOv8n": (FABRIC_ROOT / "weights" / "YOLOv8n.pt", FABRIC_ROOT / "results" / "yolov8n_metrics.json"),
        "YOLOv8s": (FABRIC_ROOT / "weights" / "YOLOv8s.pt", FABRIC_ROOT / "results" / "yolov8s_metrics.json"),
        "YOLOv11n": (FABRIC_ROOT / "weights" / "YOLOv11.pt", FABRIC_ROOT / "results" / "yolov11n_metrics.json"),
    }
    for model_key, (w_path, m_path) in yolo_weights.items():
        if not w_path.exists() or not m_path.exists():
            print(f"[SKIP] {model_key}: weights or metrics missing")
            continue
        print(f"Measuring {model_key} ...")
        accuracy = _parse_yolo_metrics(m_path)
        stats = _measure_yolo_model(model_key, w_path, accuracy, device)
        results.append(stats)
        print(f"  {stats.total_params_m:.1f}M params, {stats.latency_ms_mean:.1f}ms/{stats.fps:.1f}fps")

    out_path = OUT_DIR / "comparison.json"
    out_path.write_text(json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False))
    print(f"\n[SUCCESS] Wrote {out_path}")

    print("\n" + "=" * 100)
    print(f"{'Model':<20} | {'Task':<16} | {'Params(M)':>10} | {'Latency(ms)':>12} | {'FPS':>8} | Accuracy")
    print("-" * 100)
    for r in results:
        acc_str = ", ".join(f"{k}={v:.1f}%" for k, v in r.accuracy.items())
        print(f"{r.name:<20} | {r.task:<16} | {r.total_params_m:>10.1f} | {r.latency_ms_mean:>12.1f} | {r.fps:>8.1f} | {acc_str}")
    print("=" * 100)


if __name__ == "__main__":
    main()
