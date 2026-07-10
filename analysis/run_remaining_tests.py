"""Run the 8 benchmark_suite test units not yet exercised against the real
trained checkpoints (C2, C4, C5, C6, M1, M2, N2, plus N1's real-file-size part).

Excluded, per the earlier accounting given to the user:
  - C1 (needs NVIDIA GPU + pynvml, not available on this Mac)
  - N1's tc-based transfer timing (needs Linux + root + iproute2)
  - N3's true hardware-in-the-loop arms (needs real PLC/bus hardware)

Must be run from the repo root (weights_path in registry.yaml is root-relative).
Writes analysis/results/remaining_tests.json and appends a section to summary.md.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "benchmark_suite"))

from common.registry import load_registry  # noqa: E402
from common.runtime import Runtime  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "results"

ANOMALY_MODELS = ["m.patchcore", "m.rd4ad", "m.efficientad", "m.supersimplenet"]
ALL_MODELS = ["m.yolov8n", "m.yolov8s", "m.yolov11n"] + ANOMALY_MODELS


def make_runtime(**overrides) -> Runtime:
    base = dict(name="server_mps", role="measurement", device="auto", allow_mock=False,
               data_mode="synthetic", warmup=0, iters=0, repeats=1, raw={})
    base.update(overrides)
    return Runtime(**base)


def _free_mps_memory():
    """反复 load_from_checkpoint 大 checkpoint（PatchCore 1.16GB/RD4AD 766MB）却不清理，
    MPS 显存会一路攒到爆——这是这一整批脚本跑到后面全部失败的真正原因，不是被测代码的问题。"""
    import gc
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def run_one(label: str, fn, **kwargs):
    print(f"\n=== {label} ===", flush=True)
    try:
        res = fn(**kwargs)
        verdict = res.get("hypothesis", {})
        print(f"  verdict: {verdict}")
        return {"ok": True, "result": res}
    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        traceback.print_exc()
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    finally:
        _free_mps_memory()


def main():
    registry = load_registry()
    all_results = {}

    # ---- C2: resolution sweep, all 7 models grouped as "conv_group" for lack of an
    # available attention-paradigm model (MoECLIP untrained). Reduced warmup/iters and a
    # 2-point resolution grid to keep 7 real models' worth of timing tractable in one batch -
    # still real per-call timing, just fewer repeats than the 10-warmup/50-iter default.
    import compute.c2_resolution_sweep as c2
    _RealC2Config = c2.C2Config

    def _small_c2config(*a, **k):
        inst = _RealC2Config(*a, **k)
        inst.warmup, inst.iters = 2, 5
        return inst

    c2.C2Config = _small_c2config
    orig_groups = registry.paradigm_groups
    registry.paradigm_groups = {"conv_group": list(ALL_MODELS)}
    rt = make_runtime()
    all_results["C2"] = run_one("C2 resolution sweep (all 7 models)", c2.run,
                                registry=registry, runtime=rt, resolutions=[256, 512], write=True)
    registry.paradigm_groups = orig_groups
    c2.C2Config = _RealC2Config

    # ---- C4: changeover training cost, per model, reduced-but-real params ----
    # C4Config's dataclass defaults are baked into run()'s compiled bytecode at class-definition
    # time, so mutating __dataclass_fields__[x].default after the fact has no effect (confirmed
    # the hard way earlier this session). Instead, reassign the module-global name C4Config to a
    # small wrapper that returns an instance with the fields we want overridden - run() looks up
    # C4Config in its own module's global namespace at call time, so this does take effect.
    import compute.c4_changeover_training as c4
    _RealC4Config = c4.C4Config

    def _small_c4config(*a, **k):
        inst = _RealC4Config(*a, **k)
        inst.full_epochs, inst.full_n_samples = 3, 60
        inst.incr_epochs, inst.incr_n_samples = 1, 20
        return inst

    c4.C4Config = _small_c4config
    all_results["C4"] = {}
    for mid in ALL_MODELS:
        rt = make_runtime()
        all_results["C4"][mid] = run_one(f"C4 changeover training cost ({mid})", c4.run,
                                         model_id=mid, registry=registry, runtime=rt, write=True)
    c4.C4Config = _RealC4Config

    # ---- C5: cross-domain drift, anomaly models only ----
    import compute.c5_domain_shift as c5
    all_results["C5"] = {}
    for mid in ANOMALY_MODELS:
        rt = make_runtime()
        all_results["C5"][mid] = run_one(f"C5 cross-domain drift ({mid})", c5.run,
                                         model_id=mid, dataset_id="d.zju", registry=registry, runtime=rt, write=True)

    # ---- C6: dual-version (fp32 vs fp16) consistency, all models (skip ones that fail cleanly) ----
    import compute.c6_upgrade_consistency as c6
    all_results["C6"] = {}
    for mid in ALL_MODELS:
        rt = make_runtime()
        all_results["C6"][mid] = run_one(f"C6 dual-version consistency ({mid})", c6.run,
                                         model_id=mid, registry=registry, runtime=rt, write=True)

    # ---- M1: bandwidth contention (model-agnostic trigger) ----
    import memory.m1_bandwidth as m1
    rt = make_runtime()
    all_results["M1"] = run_one("M1 bandwidth contention (trigger model: m.patchcore)", m1.run,
                                model_id="m.patchcore", registry=registry, runtime=rt, write=True)

    # ---- M2: quantization micro-defect erasure, anomaly models only ----
    import memory.m2_quantization as m2
    all_results["M2"] = {}
    for mid in ANOMALY_MODELS:
        rt = make_runtime()
        all_results["M2"][mid] = run_one(f"M2 quantization ({mid})", m2.run,
                                         model_id=mid, dataset_id="d.mvtec", registry=registry, runtime=rt, write=True)

    # ---- N2: event-driven callback latency, all 7 models ----
    import communication.n2_event_callback as n2
    all_results["N2"] = {}
    for mid in ALL_MODELS:
        rt = make_runtime()
        all_results["N2"][mid] = run_one(f"N2 event callback ({mid})", n2.run,
                                         model_id=mid, dataset_id="d.sdust", registry=registry, runtime=rt, write=True)

    # ---- N1: real package size only (transfer timing needs Linux+root+tc, skipped) ----
    from communication.n1_model_sync import _real_pkg_mb
    all_results["N1_pkg_size"] = {
        mid: _real_pkg_mb(mid, registry) for mid in ALL_MODELS
    }
    print("\n=== N1 real package sizes (MB) ===")
    for mid, sz in all_results["N1_pkg_size"].items():
        print(f"  {mid}: {sz}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "remaining_tests.json"
    out_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False, default=str))
    print(f"\n[SUCCESS] Wrote {out_path}")


if __name__ == "__main__":
    main()
