"""N1 —— 分布式版本同步：模型分发与同步测试（Handbook §3 / 假设 H9）。

目的：量化模型更新下发到多工位的耗时与带宽冲击，验证异常检测模型同步开销是否高于监督模型。
判据（prior）：异常检测模型（含 memory bank，包体积大）fleet 同步耗时 高于 监督模型 → H9 支持。

执行模型（§2.8）：
  - server（真实）：tc 模拟网络，起 N 节点 + 中心服务器，打点下载/校验/加载各阶段时间。
  - local（可行性）：用各模型包体积 + 网络带宽合成同步耗时，演示异常/监督差异。

可单测核心逻辑：fleet_sync_time / judge_h9。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from common.registry import load_registry
from common.runtime import Runtime, load_runtime
from common.result import build_result, write_result, validate_result

# 各模型部署包体积（MB，名义）：PatchCore memory bank 使包远大于权重
_PKG_MB = {"m.yolov8n": 6, "m.yolov8s": 22, "m.yolov11n": 6,
           "m.patchcore": 210, "m.rd4ad": 90, "m.efficientad": 30, "m.moeclip": 1700}

# 网络场景：带宽 Mbps、节点数、并发度（同时下发的节点数）
NETWORKS = {
    "lan":      {"bw_mbps": 1000, "nodes": 10,  "parallel": 10},
    "factory":  {"bw_mbps": 100,  "nodes": 50,  "parallel": 10, "loss": 0.01},
    "edge_wan": {"bw_mbps": 10,   "nodes": 100, "parallel": 5},
}


@dataclass
class N1Config:
    networks: Dict[str, dict] = field(default_factory=lambda: dict(NETWORKS))


# --------------------------------------------------------------------------- #
def fleet_sync_time(pkg_mb: float, bw_mbps: float, nodes: int, parallel: int,
                    loss: float = 0.0) -> float:
    """N 节点全部完成更新的时间（秒）。分批并发下发。"""
    per_device_s = (pkg_mb * 8) / bw_mbps                 # MB→Mbit / Mbps
    per_device_s *= (1 + loss * 3)                        # 丢包重传的粗略惩罚
    batches = -(-nodes // max(1, parallel))               # ceil
    return per_device_s * batches


def judge_h9(anomaly_sync: List[float], supervised_sync: List[float]) -> Tuple[str, str]:
    if not anomaly_sync or not supervised_sync:
        return "unfilled", "缺少分组同步数据"
    a = sum(anomaly_sync) / len(anomaly_sync)
    s = sum(supervised_sync) / len(supervised_sync)
    ev = f"异常检测组均值 {a:.0f}s vs 监督组 {s:.0f}s"
    if a > s:
        return "supported", ev + "（异常同步开销更高）"
    return "not_supported", ev


# --------------------------------------------------------------------------- #
def run(model_id: str | None = None, dataset_id: str = "any",
        registry=None, runtime: Runtime | None = None, write: bool | None = None) -> Dict[str, Any]:
    registry = registry or load_registry()
    rt = runtime or load_runtime("local")
    cfg = N1Config()

    sup = [m for m, s in registry.models.items() if s.get("task") == "detection"]
    ano = [m for m, s in registry.models.items() if s.get("paradigm") == "feature_embed"]
    targets = (sup + ano) if model_id is None else [model_id]

    per_model: Dict[str, Any] = {}
    sync_worst: Dict[str, float] = {}      # 取最差网络(edge_wan)的 fleet 同步耗时代表
    for mid in targets:
        pkg = _PKG_MB.get(mid, 50)
        by_net = {}
        for net, p in cfg.networks.items():
            t = fleet_sync_time(pkg, p["bw_mbps"], p["nodes"], p["parallel"], p.get("loss", 0.0))
            by_net[net] = {"fleet_sync_s": t, "per_device_s": (pkg * 8) / p["bw_mbps"]}
        per_model[mid] = {"pkg_mb": pkg, "task": registry.models[mid].get("task"), "by_network": by_net}
        sync_worst[mid] = by_net["edge_wan"]["fleet_sync_s"]

    verdict, evidence = judge_h9([sync_worst[m] for m in ano if m in sync_worst],
                                 [sync_worst[m] for m in sup if m in sync_worst])

    res = build_result(
        test_id="N1", test_name="model_distribution_sync",
        measurement_type="real" if rt.is_measurement else "proxy", hardware_id="hw.cloud",
        config={"networks": cfg.networks, "targets": targets, "mode": rt.name},
        metrics={"per_model": per_model, "fleet_sync_worst_s": sync_worst,
                 "summary_lines": [f"{m}: 包 {per_model[m]['pkg_mb']}MB, edge_wan 同步 {sync_worst[m]:.0f}s"
                                   for m in sync_worst]},
        hypothesis_id="H9", verdict=verdict, evidence=evidence,
        notes=("local 可行性：包体积×带宽合成同步耗时，非真实 tc 网络。" if rt.is_feasibility else ""),
    )
    if validate_result(res):
        raise RuntimeError("result 不合规")
    if (rt.is_measurement if write is None else write):
        res["_written_to"] = str(write_result(res))
    return res


if __name__ == "__main__":
    import json
    r = run()
    print(json.dumps({"verdict": r["hypothesis"], "sync": r["metrics"]["fleet_sync_worst_s"]},
                     indent=2, ensure_ascii=False))
