"""模型适配层 —— 把三个来源仓库统一到同一接口（Handbook §2.2 复用策略）。

统一接口（每个适配器都实现）：
    load()                -> None          # 载入真实模型或（可行性下）mock
    preprocess(img)       -> tensor/array  # 归一化/缩放到模型输入
    infer(x)              -> ModelOutput    # 单次前向
    export(fmt, path)     -> path          # onnx/fp16/int8 导出（真实测量时用）

设计要点：
  - 适配器不重写模型，只做"套壳"：ultralytics 走 YOLO，anomaly 走仓库 pipeline，MoECLIP 走其 model 包。
  - 可行性模式(runtime.is_feasibility)下，若真实库/权重不可用，回退到 MockModel，
    以便本地在无 GPU/无权重时也能走通 load→preprocess→infer 的装配链路。
  - 服务器测量模式(allow_mock=False)下缺依赖会抛错，不静默 mock。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from .runtime import Runtime


@dataclass
class ModelOutput:
    """统一输出容器。检测→boxes；异常→score/anomaly_map。可行性下为占位。"""
    raw: Any = None
    backend: str = "mock"          # real | mock
    shape: Optional[tuple] = None
    meta: Dict[str, Any] = field(default_factory=dict)


class MockModel:
    """占位模型：返回形状正确的假输出，用于本地可行性验证。"""
    def __init__(self, model_id: str, task: str, input_hw: int):
        self.model_id = model_id
        self.task = task
        self.input_hw = input_hw

    def __call__(self, x):
        # 不依赖 torch：只报告输入形状，产出占位输出
        shp = tuple(getattr(x, "shape", ()) or ())
        return ModelOutput(raw={"mock": True}, backend="mock", shape=shp,
                           meta={"task": self.task, "model_id": self.model_id})


class BaseModelAdapter:
    adapter_key: str = "base"

    def __init__(self, model_id: str, spec: Dict[str, Any], runtime: Runtime):
        self.model_id = model_id
        self.spec = spec
        self.rt = runtime
        self.task = spec.get("task")
        self.input_hw = int(spec.get("default_input", 256))
        self.model = None
        self.backend = "mock"

    # 子类覆盖：尝试载入真实模型；失败且可行性→mock
    def _load_real(self):
        raise NotImplementedError

    def load(self):
        try:
            self.model = self._load_real()
            self.backend = "real"
        except Exception as e:
            if self.rt.is_feasibility and self.rt.allow_mock:
                self.model = MockModel(self.model_id, self.task, self.input_hw)
                self.backend = "mock"
                self._load_note = f"mock（真实载入不可用：{type(e).__name__}）"
            else:
                raise
        return self

    def preprocess(self, img=None):
        # 可行性下：无输入则用合成图；真实路径由子类按需覆盖
        if img is None:
            return self.rt.synthetic_image(1, 3, self.input_hw, self.input_hw)
        return img

    def infer(self, x) -> ModelOutput:
        if self.model is None:
            raise RuntimeError("请先 load()")
        if isinstance(self.model, MockModel):
            return self.model(x)
        return self._infer_real(x)

    def _infer_real(self, x) -> ModelOutput:
        raise NotImplementedError

    def export(self, fmt: str, path: str):
        raise NotImplementedError(f"{self.adapter_key} 尚未实现 export({fmt})")


# --------------------------------------------------------------------------- #
# 具体适配器
# --------------------------------------------------------------------------- #
class UltralyticsAdapter(BaseModelAdapter):
    """YOLOv8/v11 —— 复用 ultralytics + fabric-defect-detection 训练出的权重。"""
    adapter_key = "ultralytics_weights"

    def _load_real(self):
        ul = self.rt.optional_import("ultralytics")
        if ul is None:
            raise ImportError("ultralytics 未安装")
        weights = self.spec.get("weights_path")
        if not weights or not Path(weights).exists():
            raise FileNotFoundError(f"权重缺失: {weights}")
        return ul.YOLO(weights)

    def _infer_real(self, x) -> ModelOutput:
        r = self.model.predict(x, verbose=False)
        return ModelOutput(raw=r, backend="real", meta={"task": self.task})


class AnomalyRepoAdapter(BaseModelAdapter):
    """PatchCore/RD4AD/EfficientAD —— 复用 anomaly_detection_for_textile_industry 仓库 pipeline。

    真实路径需将该仓库置于 PYTHONPATH 并按 config.yaml 载入；此处先保证可行性(mock)可通，
    真实 pipeline 对接留待"数据/训练"步骤（因其依赖已训练权重与 config）。
    """
    adapter_key = "repo_pipeline"

    def _load_real(self):
        # TODO(server步): from src.anomaly_pipeline import load_model; return load_model(cfg)
        raise NotImplementedError("anomaly 仓库真实对接在后续步骤实现")


class MoECLIPAdapter(BaseModelAdapter):
    """MoECLIP —— import 其 model 包并载入 OpenCLIP 权重。"""
    adapter_key = "import_model_pkg"

    def _load_real(self):
        # TODO(server步): 将 MoECLIP 置于 PYTHONPATH，from model.model import build_model
        raise NotImplementedError("MoECLIP 真实对接在后续步骤实现")


_ADAPTERS = {
    UltralyticsAdapter.adapter_key: UltralyticsAdapter,
    AnomalyRepoAdapter.adapter_key: AnomalyRepoAdapter,
    MoECLIPAdapter.adapter_key: MoECLIPAdapter,
}


def get_adapter(model_id: str, registry, runtime: Runtime) -> BaseModelAdapter:
    spec = registry.models[model_id]
    key = spec.get("adapter", "base")
    cls = _ADAPTERS.get(key)
    if cls is None:
        raise KeyError(f"未知适配器类型 '{key}'（模型 {model_id}）")
    return cls(model_id, spec, runtime)


def feasibility_check(model_id: str, registry, runtime: Runtime) -> Dict[str, Any]:
    """本地可行性验证：load→合成输入→preprocess→infer，报告是否走通及后端(real/mock)。"""
    out = {"model": model_id, "backend": None, "ok": False, "note": ""}
    try:
        ad = get_adapter(model_id, registry, runtime).load()
        x = ad.preprocess(None)
        y = ad.infer(x)
        out["backend"] = ad.backend
        out["ok"] = isinstance(y, ModelOutput)
        out["note"] = getattr(ad, "_load_note", "real 载入成功") if ad.backend == "real" \
            else getattr(ad, "_load_note", "mock")
        out["out_shape"] = y.shape
    except Exception as e:
        out["note"] = f"{type(e).__name__}: {e}"
    return out
