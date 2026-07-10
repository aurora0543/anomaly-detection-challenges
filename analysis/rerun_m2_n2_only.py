"""Targeted rerun of just M2 (4 anomaly models) and N2 (3 models that OOM'd), after
fixing the M2 device-mismatch bug and adding MPS memory cleanup between iterations.
Skips C4/C5's m.efficientad retry - that needs a ~1.5GB Imagenette download the user
chose not to wait for; those two stay marked as failed/skipped in the results.
"""
from __future__ import annotations

import gc
import json
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "benchmark_suite"))

from common.registry import load_registry  # noqa: E402
from common.runtime import Runtime  # noqa: E402

RESULTS_PATH = Path(__file__).resolve().parent / "results" / "remaining_tests.json"


def make_runtime(**overrides) -> Runtime:
    base = dict(name="server_mps", role="measurement", device="auto", allow_mock=False,
               data_mode="synthetic", warmup=0, iters=0, repeats=1, raw={})
    base.update(overrides)
    return Runtime(**base)


def _free_mps_memory():
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def run_one(label, fn, **kwargs):
    print(f"\n=== {label} ===", flush=True)
    try:
        res = fn(**kwargs)
        print(f"  verdict: {res.get('hypothesis')}")
        return {"ok": True, "result": res}
    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        traceback.print_exc()
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    finally:
        _free_mps_memory()


def main():
    registry = load_registry()
    all_results = json.loads(RESULTS_PATH.read_text())

    # mark the skipped-by-user-choice items explicitly, distinct from a real failure
    all_results["C4"]["m.efficientad"] = {
        "ok": False,
        "error": "SKIPPED: needs first-run ~1.5GB Imagenette auto-download; user chose not to wait",
    }
    all_results["C5"]["m.efficientad"] = {
        "ok": False,
        "error": "SKIPPED: needs first-run ~1.5GB Imagenette auto-download; user chose not to wait",
    }

    import memory.m2_quantization as m2
    for mid in ["m.patchcore", "m.rd4ad", "m.efficientad", "m.supersimplenet"]:
        all_results["M2"][mid] = run_one(
            f"M2 quantization ({mid}) [retry]", m2.run,
            model_id=mid, dataset_id="d.mvtec", registry=registry, runtime=make_runtime(), write=True,
        )

    import communication.n2_event_callback as n2
    for mid in ["m.patchcore", "m.efficientad", "m.supersimplenet"]:
        all_results["N2"][mid] = run_one(
            f"N2 event callback ({mid}) [retry]", n2.run,
            model_id=mid, dataset_id="d.sdust", registry=registry, runtime=make_runtime(), write=True,
        )

    RESULTS_PATH.write_text(json.dumps(all_results, indent=2, ensure_ascii=False, default=str))
    print(f"\n[SUCCESS] Updated {RESULTS_PATH}")


if __name__ == "__main__":
    main()
