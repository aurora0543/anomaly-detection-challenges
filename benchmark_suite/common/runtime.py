"""执行运行时 —— 统一管理"本地可行性 vs 服务器真实测量"两档（Handbook §2.8）。

核心职责：
  - 解析运行档（local/server），合并 measurement.yaml 的默认循环参数；
  - 决定设备（cuda/cpu）与是否允许 mock；
  - 提供合成输入生成、可选依赖导入等可行性验证所需的工具。

本地(feasibility)下：device 强制回退 cpu，缺 torch 也能用 numpy 合成输入走通装配流程。
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional

from .registry import load_yaml


@dataclass
class Runtime:
    name: str                 # local | server
    role: str                 # feasibility | measurement
    device: str               # cpu | cuda
    allow_mock: bool
    data_mode: str            # synthetic | cloud | local
    warmup: int
    iters: int
    repeats: int
    raw: dict

    @property
    def is_feasibility(self) -> bool:
        return self.role == "feasibility"

    @property
    def is_measurement(self) -> bool:
        return self.role == "measurement"

    # -- 可选依赖导入：可行性模式下缺失返回 None，测量模式下缺失抛错 --
    def optional_import(self, module_name: str):
        try:
            import importlib
            return importlib.import_module(module_name)
        except Exception:
            if self.is_measurement and not self.allow_mock:
                raise RuntimeError(
                    f"[{self.name}] 缺少依赖 '{module_name}'，真实测量模式不允许 mock。")
            return None

    # -- 有效设备：measurement 想用 cuda 但环境无 torch/cuda 时的处理 --
    # 注意：Apple Silicon 的 MPS 后端也是真实 GPU，不是"没有 GPU"——之前这里只认 cuda，
    # 在 Mac 上会被误判成"只能退化成 CPU"，这是本框架的一个真实 bug，不是设备本身的限制。
    def resolve_device(self) -> str:
        if self.device == "cpu":
            return "cpu"
        torch = self.optional_import("torch")
        if self.device == "cuda":
            if torch is not None and torch.cuda.is_available():
                return "cuda"
        elif self.device in ("gpu", "auto", "mps"):
            if torch is not None and torch.cuda.is_available():
                return "cuda"
            if torch is not None and getattr(torch.backends, "mps", None) is not None \
                    and torch.backends.mps.is_available():
                return "mps"
        if self.is_feasibility:
            return "cpu"   # 本地无 GPU：可行性验证退回 cpu
        raise RuntimeError(f"[{self.name}] 请求 GPU（{self.device}）但不可用（torch/CUDA/MPS 均缺失）。")

    # -- 合成输入：优先返回 torch 张量，无 torch 时返回 numpy，供可行性前向 --
    def synthetic_image(self, batch: int, ch: int, h: int, w: int):
        torch = self.optional_import("torch")
        if torch is not None:
            dev = self.resolve_device()
            return torch.rand(batch, ch, h, w, device=dev)
        import numpy as np
        return np.random.rand(batch, ch, h, w).astype("float32")


def load_runtime(profile: Optional[str] = None,
                 runtime_yaml: str = "runtime.yaml",
                 measurement_yaml: str = "measurement.yaml") -> Runtime:
    rt = load_yaml(runtime_yaml)
    meas = load_yaml(measurement_yaml)
    name = profile or rt.get("default_profile", "local")
    prof = rt["profiles"][name]

    def pick(key, fallback):
        v = prof.get(key)
        return fallback if v is None else v

    return Runtime(
        name=name,
        role=prof.get("role", "feasibility"),
        device=prof.get("device", "cpu"),
        allow_mock=bool(prof.get("allow_mock", name == "local")),
        data_mode=prof.get("data_mode", "synthetic"),
        warmup=pick("warmup", meas.get("warmup", 100)),
        iters=pick("iters", meas.get("iters", 1000)),
        repeats=pick("repeats", meas.get("repeats", 5)),
        raw=prof,
    )
