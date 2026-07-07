"""C1 —— 端侧资源受限：持续负载功耗-热-降频测试（Handbook §3 / 假设 H1）。

目的：验证功耗/散热受限（无风扇被动散热）下，模型能否维持产线节拍所需的持续实时性。
判据（hard）：持续吞吐 fps_sustained < 产线节拍 line_speed_fps → H1 成立（实时性无法保障）。

执行模型（§2.8）：
  - server/edge（真实）：设功耗档或环境温度，持续 30min 推理，后台采样 power/temp/clock + 每帧延迟。
  - local（可行性）：用热降频模型合成 fps/temp 时序，演示"持续吞吐跌破节拍"。

可单测核心逻辑：analyze_fps_series / judge_h1。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from common.registry import load_registry
from common.runtime import Runtime, load_runtime
from common.models import get_adapter
from common.result import build_result, write_result, validate_result

LINE_SPEED_FPS_DEFAULT = 30.0            # 产线节拍（实例：≥30 FPS）——hard 门槛，运行前锁定
POWER_CAPS_W = [30, 50, 70]

# 复杂度基准 FPS（满功耗、ref 条件），与 C2 一致口径
_BASE_FPS = {"conv": 200.0, "feature_embed": 80.0, "attention": 40.0}
# 被动散热降频参数（local 合成）：稳态占峰值比例、降频起始秒
_PASSIVE_SUSTAIN_FRAC = 0.55
_THROTTLE_ONSET_S = 90.0


@dataclass
class C1Config:
    duration_s: int = 1800
    sample_hz: float = 1.0
    power_cap_w: int = 50
    max_power_w: int = 100
    line_speed_fps: float = LINE_SPEED_FPS_DEFAULT
    resolution: int = 512


# --------------------------------------------------------------------------- #
# 可单测核心逻辑
# --------------------------------------------------------------------------- #
def analyze_fps_series(times: List[float], fps: List[float]) -> Dict[str, float]:
    """从 fps 时序提取 peak / sustained / decay_pct / throttle_onset_s。"""
    if not fps:
        return {}
    peak = max(fps[:max(1, len(fps) // 10)])          # 前 10% 的峰值
    tail = fps[int(len(fps) * 0.8):] or fps[-1:]      # 末 20% 稳态
    sustained = sum(tail) / len(tail)
    decay_pct = (peak - sustained) / peak if peak else float("nan")
    onset = float("nan")
    for t, f in zip(times, fps):
        if f < 0.95 * peak:
            onset = t
            break
    return {"fps_peak": peak, "fps_sustained": sustained,
            "fps_decay_pct": decay_pct, "throttle_onset_s": onset}


def judge_h1(fps_sustained: float, line_speed_fps: float) -> Tuple[str, str]:
    if fps_sustained != fps_sustained:
        return "unfilled", "无稳态吞吐数据"
    if fps_sustained < line_speed_fps:
        return "supported", f"持续 {fps_sustained:.1f} FPS < 节拍 {line_speed_fps:.0f} → 实时性无法保障"
    return "not_supported", f"持续 {fps_sustained:.1f} FPS ≥ 节拍 {line_speed_fps:.0f}"


# --------------------------------------------------------------------------- #
def _synthetic_series(adapter, cfg: C1Config) -> Tuple[List[float], List[float], List[float]]:
    """合成 (times, fps, temp)：受功耗封顶压制峰值 + 被动散热降频。"""
    base = _BASE_FPS.get(adapter.spec.get("paradigm"), 100.0)
    peak = base * (cfg.power_cap_w / cfg.max_power_w)          # 功耗档压制峰值
    n = max(2, int(cfg.duration_s * cfg.sample_hz))
    times, fps, temp = [], [], []
    for i in range(n):
        t = i / cfg.sample_hz
        if t < _THROTTLE_ONSET_S:
            f = peak
            tp = 45 + (t / _THROTTLE_ONSET_S) * 40             # 升温到 ~85
        else:
            f = peak * _PASSIVE_SUSTAIN_FRAC                    # 降频后稳态
            tp = 85.0
        times.append(t); fps.append(f); temp.append(tp)
    return times, fps, temp


def run(model_id: str = "m.patchcore", dataset_id: str = "d.mvtec",
        registry=None, runtime: Runtime | None = None,
        line_speed_fps: float | None = None, write: bool | None = None) -> Dict[str, Any]:
    registry = registry or load_registry()
    rt = runtime or load_runtime("local")
    model_id = model_id or "m.patchcore"
    cfg = C1Config(line_speed_fps=line_speed_fps or LINE_SPEED_FPS_DEFAULT)
    adapter = get_adapter(model_id, registry, rt).load()

    if rt.is_measurement and adapter.backend == "real":
        # TODO(server/edge): power.py 后台采样 + 持续推理循环，得到真实 times/fps/temp
        raise NotImplementedError("C1 真实功耗-热采样在 power.py + 边缘/proxy 步骤对接")
    times, fps, temp = _synthetic_series(adapter, cfg)

    stats = analyze_fps_series(times, fps)
    verdict, evidence = judge_h1(stats.get("fps_sustained", float("nan")), cfg.line_speed_fps)

    measurement_type = "real" if rt.is_measurement else "proxy"
    res = build_result(
        test_id="C1", test_name="power_thermal_throttle",
        measurement_type=measurement_type, hardware_id="hw.edge",
        config={"model_id": model_id, "power_cap_w": cfg.power_cap_w,
                "duration_s": cfg.duration_s, "line_speed_fps": cfg.line_speed_fps,
                "mode": rt.name, "backend": adapter.backend},
        metrics={**stats, "temp_max_c": max(temp) if temp else None,
                 "summary_lines": [
                     f"峰值 {stats.get('fps_peak', 0):.1f} FPS → 稳态 {stats.get('fps_sustained', 0):.1f} FPS "
                     f"(降频 {stats.get('fps_decay_pct', 0)*100:.0f}%)",
                     f"首次降频 @ {stats.get('throttle_onset_s')}s，峰值温度 {max(temp):.0f}°C",
                 ]},
        hypothesis_id="H1", verdict=verdict, evidence=evidence,
        notes=("local 可行性：功耗封顶+被动散热的合成时序，非真实端侧测量。"
               if rt.is_feasibility else ""),
    )
    if validate_result(res):
        raise RuntimeError("result 不合规")
    if (rt.is_measurement if write is None else write):
        res["_written_to"] = str(write_result(res))
    return res


if __name__ == "__main__":
    import json
    r = run()
    print(json.dumps({"verdict": r["hypothesis"], "metrics": {k: v for k, v in r["metrics"].items()
                                                              if k != "summary_lines"}},
                     indent=2, ensure_ascii=False))
