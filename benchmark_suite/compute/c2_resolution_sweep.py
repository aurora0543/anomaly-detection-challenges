"""C2 —— 注意力复杂度：分辨率-吞吐扫频测试（Handbook §3 / 假设 H2）。

目的：验证含注意力骨干的模型在分辨率升高时吞吐下降是否显著快于纯卷积模型（O(n²) vs O(n)）。
判据：注意力组 吞吐-分辨率 log-log 斜率 比卷积组更陡（更负），且分组可分 → H2 支持。

执行模型（§2.8）：
  - server（真实测量）：对每个模型、每个分辨率用合成/真实输入 warmup+iters 计时，得 FPS。
  - local（可行性）：用复杂度模型合成 FPS —— 注意力计算约 O(res^4)（token 数∝(res/patch)²，
    注意力 O(token²)），卷积约 O(res^2)；据此演示 H2 模式，不写 result.json。

可单测核心逻辑：loglog_slope / slope_gap / judge_h2（与模型无关）。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from common.registry import load_registry
from common.runtime import Runtime, load_runtime
from common.models import get_adapter
from common.bench_utils import run_timed, summarize
from common.result import build_result, write_result, validate_result

RESOLUTIONS = [256, 512, 1024, 2048]

# 复杂度模型（local 演示用）：FPS ∝ 1/(res/ref)^exp
_REF_RES = 256
_PARADIGM_EXP = {"conv": 2.0, "feature_embed": 2.0, "attention": 3.8}   # 注意力更陡
_BASE_FPS = {"conv": 200.0, "feature_embed": 80.0, "attention": 40.0}   # ref 分辨率处名义 FPS
_BASE_GFLOPS = {"conv": 8.0, "feature_embed": 20.0, "attention": 80.0}


@dataclass
class C2Config:
    dataset_id: str
    resolutions: List[int] = field(default_factory=lambda: list(RESOLUTIONS))
    warmup: int = 10
    iters: int = 50
    batch_size: int = 1


# --------------------------------------------------------------------------- #
# 可单测核心逻辑
# --------------------------------------------------------------------------- #
def loglog_slope(resolutions: List[int], fps: List[float]) -> float:
    """log(FPS) 对 log(res) 的线性拟合斜率（越负=随分辨率下降越快）。"""
    import math
    xs = [math.log(r) for r in resolutions]
    ys = [math.log(f) for f in fps]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den else float("nan")


def judge_h2(attn_slopes: List[float], conv_slopes: List[float]) -> Tuple[str, str, float]:
    """注意力更陡（更负）且两组可分 → supported。返回 (verdict, evidence, slope_gap)。"""
    if not attn_slopes or not conv_slopes:
        return "unfilled", "缺少分组斜率", float("nan")
    attn_mean = sum(attn_slopes) / len(attn_slopes)
    conv_mean = sum(conv_slopes) / len(conv_slopes)
    gap = conv_mean - attn_mean          # >0 表示注意力更负（更陡）
    separated = max(attn_slopes) < min(conv_slopes)
    ev = f"注意力斜率均值{attn_mean:.2f} vs 卷积{conv_mean:.2f}, gap={gap:.2f}"
    if gap > 0 and separated:
        return "supported", ev + "（两组可分）", gap
    if gap > 0:
        return "partial", ev + "（方向对但两组有重叠）", gap
    return "not_supported", ev, gap


# --------------------------------------------------------------------------- #
# FPS / GFLOPs 测量：真实（server）/ 合成（local）
# --------------------------------------------------------------------------- #
def _measure_fps(adapter, res: int, cfg: C2Config, rt: Runtime) -> float:
    if rt.is_measurement and adapter.backend == "real":
        x = adapter.preprocess(rt.synthetic_image(cfg.batch_size, 3, res, res))
        lat = run_timed(lambda: adapter.infer(x), warmup=cfg.warmup, iters=cfg.iters)
        return summarize(lat)["fps"]
    # 合成复杂度模型
    exp = _PARADIGM_EXP.get(adapter.spec.get("paradigm"), 2.0)
    base = _BASE_FPS.get(adapter.spec.get("paradigm"), 100.0)
    return base / ((res / _REF_RES) ** exp)


def _gflops(adapter, res: int, rt: Runtime) -> float:
    exp = _PARADIGM_EXP.get(adapter.spec.get("paradigm"), 2.0)
    base = _BASE_GFLOPS.get(adapter.spec.get("paradigm"), 10.0)
    return base * ((res / _REF_RES) ** exp)   # server 可用 thop 覆盖


# --------------------------------------------------------------------------- #
# 编排入口（多模型分组）
# --------------------------------------------------------------------------- #
def run(model_id: str | None = None, dataset_id: str = "d.mvtec",
        registry=None, runtime: Runtime | None = None,
        resolutions: List[int] | None = None, write: bool | None = None) -> Dict[str, Any]:
    registry = registry or load_registry()
    rt = runtime or load_runtime("local")
    cfg = C2Config(dataset_id=dataset_id, resolutions=resolutions or list(RESOLUTIONS))

    groups = registry.paradigm_groups   # attention_group / conv_group
    per_model: Dict[str, Any] = {}
    slopes: Dict[str, float] = {}
    for gname, mids in groups.items():
        for mid in mids:
            adapter = get_adapter(mid, registry, rt).load()
            fps_curve, gflops_curve = [], []
            for res in cfg.resolutions:
                fps_curve.append(_measure_fps(adapter, res, cfg, rt))
                gflops_curve.append(_gflops(adapter, res, rt))
            sl = loglog_slope(cfg.resolutions, fps_curve)
            slopes[mid] = sl
            per_model[mid] = {"group": gname, "backend": adapter.backend,
                              "resolutions": cfg.resolutions,
                              "fps": fps_curve, "gflops": gflops_curve, "slope": sl}

    attn = [slopes[m] for m in groups.get("attention_group", []) if m in slopes]
    conv = [slopes[m] for m in groups.get("conv_group", []) if m in slopes]
    verdict, evidence, gap = judge_h2(attn, conv)

    def _grp(v):
        return {"slopes": v, "mean": (sum(v) / len(v)) if v else None,
                "note": "n<2，暂无 CI" if len(v) < 2 else "n≥2 可 bootstrap CI"}

    measurement_type = "real" if rt.is_measurement else "proxy"
    res = build_result(
        test_id="C2", test_name="resolution_throughput_sweep",
        measurement_type=measurement_type, hardware_id="hw.cloud",
        config={"dataset_id": dataset_id, "resolutions": cfg.resolutions,
                "warmup": cfg.warmup, "iters": cfg.iters,
                "mode": rt.name, "groups": {k: v for k, v in groups.items()}},
        metrics={"per_model": per_model,
                 "slope_by_group": {"attention": _grp(attn), "conv": _grp(conv)},
                 "slope_gap": {"value": gap}},
        hypothesis_id="H2", verdict=verdict, evidence=evidence,
        notes=("local 可行性：FPS 来自复杂度模型（注意力 O(res^4) vs 卷积 O(res^2)），非真实计时。"
               if rt.is_feasibility else ""),
    )
    issues = validate_result(res)
    if issues:
        raise RuntimeError("result 不合规: " + "; ".join(issues))
    do_write = rt.is_measurement if write is None else write
    if do_write:
        res["_written_to"] = str(write_result(res))
    return res


if __name__ == "__main__":
    import json
    r = run()
    print(json.dumps({"verdict": r["hypothesis"],
                      "slopes": {m: round(v["slope"], 2) for m, v in r["metrics"]["per_model"].items()}},
                     indent=2, ensure_ascii=False))
