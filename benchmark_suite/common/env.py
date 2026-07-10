"""collect_env() —— 采集可复现所需的硬件/软件环境元数据（Handbook §2.6 强制字段）。

设计原则：所有第三方依赖（torch/pynvml/psutil）都是可选的，缺失时降级为 None，
以便在没有 GPU/torch 的环境（如纯 CPU 沙盒）也能运行注册表校验与自测。
"""
from __future__ import annotations
import platform
import os
from datetime import datetime, timezone


def _try(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def _gpu_info():
    info = {"gpu": None, "driver": None, "cuda": None, "gpu_count": 0}
    # 优先 pynvml（不依赖 torch）
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        info["gpu_count"] = pynvml.nvmlDeviceGetCount()
        if info["gpu_count"] > 0:
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(h)
            info["gpu"] = name.decode() if isinstance(name, bytes) else name
        info["driver"] = _try(lambda: pynvml.nvmlSystemGetDriverVersion().decode()
                              if isinstance(pynvml.nvmlSystemGetDriverVersion(), bytes)
                              else pynvml.nvmlSystemGetDriverVersion())
        pynvml.nvmlShutdown()
    except Exception:
        pass
    # cuda 版本从 torch 取（若有）
    try:
        import torch  # type: ignore
        info["cuda"] = torch.version.cuda
        if not info["gpu"] and torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
        elif not info["gpu"] and getattr(torch.backends, "mps", None) is not None \
                and torch.backends.mps.is_available():
            # Apple Silicon 的 MPS 也是真实 GPU，不是"没有 GPU"——pynvml/CUDA 都探测不到时
            # 补上这一档，避免 collect_env() 在 Mac 上把 GPU 报告成 None。
            info["gpu"] = f"Apple Silicon (MPS) - {platform.machine()}"
            info["gpu_count"] = 1
    except Exception:
        pass
    return info


def _pkg_version(name):
    try:
        import importlib.metadata as m
        return m.version(name)
    except Exception:
        return None


def collect_env() -> dict:
    ram_gb = None
    try:
        import psutil  # type: ignore
        ram_gb = round(psutil.virtual_memory().total / 1024**3, 1)
    except Exception:
        pass

    env = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "ram_gb": ram_gb,
        **_gpu_info(),
        "torch": _pkg_version("torch"),
        "ultralytics": _pkg_version("ultralytics"),
        "onnxruntime": _pkg_version("onnxruntime"),
    }
    return env


if __name__ == "__main__":
    import json
    print(json.dumps(collect_env(), indent=2, ensure_ascii=False))
