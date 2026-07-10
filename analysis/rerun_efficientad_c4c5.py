"""Retry C4 and C5 for EfficientAD now that datasets/imagenette/imagenette2.tgz has
been placed manually (network to S3 was too unstable for any downloader, including
anomalib's own, to complete in reasonable time; user supplied the file directly).
Same reduced-but-real params as the original run_remaining_tests.py batch, so the
numbers are comparable to the other 6 models' C4/C5 entries already in remaining_tests.json.
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

    import compute.c4_changeover_training as c4
    _RealC4Config = c4.C4Config

    def _small_c4config(*a, **k):
        inst = _RealC4Config(*a, **k)
        inst.full_epochs, inst.full_n_samples = 3, 60
        inst.incr_epochs, inst.incr_n_samples = 1, 20
        return inst

    c4.C4Config = _small_c4config
    all_results["C4"]["m.efficientad"] = run_one(
        "C4 changeover training cost (m.efficientad) [retry]", c4.run,
        model_id="m.efficientad", registry=registry, runtime=make_runtime(), write=True,
    )
    c4.C4Config = _RealC4Config

    import compute.c5_domain_shift as c5
    all_results["C5"]["m.efficientad"] = run_one(
        "C5 cross-domain drift (m.efficientad) [retry]", c5.run,
        model_id="m.efficientad", dataset_id="d.zju", registry=registry, runtime=make_runtime(), write=True,
    )

    RESULTS_PATH.write_text(json.dumps(all_results, indent=2, ensure_ascii=False, default=str))
    print(f"\n[SUCCESS] Updated {RESULTS_PATH}")


if __name__ == "__main__":
    main()
