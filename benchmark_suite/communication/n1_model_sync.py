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


# --------------------------------------------------------------------------- #
# 真实包体积：直接读 registry.yaml weights_path 指向的真实文件大小，替换写死的 _PKG_MB。
# 这部分在任何机器上都是真实可测的，不需要网络设备。
# --------------------------------------------------------------------------- #
def _real_pkg_mb(model_id: str, registry) -> float | None:
    from pathlib import Path
    weights = registry.models.get(model_id, {}).get("weights_path")
    if not weights or not Path(weights).exists():
        return None
    return Path(weights).stat().st_size / (1024 ** 2)


# --------------------------------------------------------------------------- #
# 真实网络同步耗时：用 Linux tc netem 在回环接口上限速/加时延，起一个本地 HTTP 服务器
# 提供真实权重文件，N 个并发客户端真实下载并计时。需要 Linux + root + tc/iproute2，
# 这台开发机（macOS）没有 tc，这条路径写好了但没法在本机端到端验证。
# --------------------------------------------------------------------------- #
def _tc_available() -> bool:
    import platform
    import shutil
    import subprocess
    if platform.system() != "Linux":
        return False
    if shutil.which("tc") is None:
        return False
    try:
        # 需要 root/NET_ADMIN 才能真的改 qdisc；用 --version 只探测命令存在与可执行
        subprocess.run(["tc", "-V"], capture_output=True, timeout=5, check=False)
        return True
    except Exception:
        return False


def _real_fleet_sync(pkg_path: str, bw_mbps: float, nodes: int, parallel: int) -> float:
    """在回环接口上用 tc netem 限速到 bw_mbps，起本地 HTTP 服务器提供 pkg_path，
    分批（每批 parallel 个）并发真实下载，返回全部节点下载完成的总耗时（秒）。"""
    import http.server
    import subprocess
    import threading
    import time
    import urllib.request
    import os
    from pathlib import Path

    if not _tc_available():
        raise RuntimeError(
            "N1 真实网络同步测量需要 Linux + root 权限 + iproute2(tc) 来限速回环接口。"
            "本机（非 Linux 或缺 tc）无法产出真实数字——请在真实 GPU/边缘服务器上以 "
            "--mode server 重跑此测试单元。"
        )

    iface, rate_kbit = "lo", int(bw_mbps * 1000)
    subprocess.run(["tc", "qdisc", "add", "dev", iface, "root", "tbf",
                   "rate", f"{rate_kbit}kbit", "burst", "32kbit", "latency", "400ms"], check=True)
    try:
        directory = str(Path(pkg_path).parent)
        filename = Path(pkg_path).name
        handler = lambda *a, **k: http.server.SimpleHTTPRequestHandler(*a, directory=directory, **k)
        httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        port = httpd.server_address[1]
        server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        server_thread.start()

        def _download():
            urllib.request.urlretrieve(f"http://127.0.0.1:{port}/{filename}", os.devnull if os.name != "nt" else "NUL")

        t0 = time.perf_counter()
        remaining = nodes
        while remaining > 0:
            batch = min(parallel, remaining)
            threads = [threading.Thread(target=_download) for _ in range(batch)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            remaining -= batch
        total_s = time.perf_counter() - t0

        httpd.shutdown()
        return total_s
    finally:
        subprocess.run(["tc", "qdisc", "del", "dev", iface, "root"], check=False)


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
        real_pkg = _real_pkg_mb(mid, registry)
        pkg = real_pkg if real_pkg is not None else _PKG_MB.get(mid, 50)
        pkg_source = "real_file" if real_pkg is not None else "nominal"

        by_net = {}
        for net, p in cfg.networks.items():
            if rt.is_measurement and real_pkg is not None:
                weights_path = registry.models[mid]["weights_path"]
                t = _real_fleet_sync(weights_path, p["bw_mbps"], p["nodes"], p["parallel"])
            else:
                t = fleet_sync_time(pkg, p["bw_mbps"], p["nodes"], p["parallel"], p.get("loss", 0.0))
            by_net[net] = {"fleet_sync_s": t, "per_device_s": (pkg * 8) / p["bw_mbps"]}
        per_model[mid] = {"pkg_mb": pkg, "pkg_source": pkg_source,
                          "task": registry.models[mid].get("task"), "by_network": by_net}
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
        notes=("local 可行性：包体积×带宽合成同步耗时，非真实 tc 网络。" if rt.is_feasibility
              else "server 真实：包体积读真实权重文件大小；同步耗时需 Linux+root+tc 在回环接口"
                   "限速后真实 HTTP 传输计时（本机缺权重文件或缺 tc 时用公式估算并在 pkg_source 标注）。"),
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
