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
        model = ul.YOLO(weights)
        model.to(self.rt.resolve_device())   # 否则默认停在 CPU，测不出真实 GPU/MPS 速度
        return model

    def _infer_real(self, x) -> ModelOutput:
        r = self.model.predict(x, verbose=False, device=self.rt.resolve_device())
        return ModelOutput(raw=r, backend="real", meta={"task": self.task})


class AnomalyRepoAdapter(BaseModelAdapter):
    """PatchCore/RD4AD/EfficientAD —— 直接用 anomalib 的 Lightning checkpoint 载入接口。

    anomalib 的模型基类在 __init__ 里调用了 save_hyperparameters()，所以训练时用过的
    backbone/layers/imagenet_dir 等构造参数都随 checkpoint 一起存了下来 ——
    直接 `<ModelClass>.load_from_checkpoint(weights_path)` 就能重建完整模型，
    不需要重新读 config.yaml 猜构造参数。

    真实前向调用的是 lightning_module.model（即 PatchcoreModel/ReverseDistillationModel/
    EfficientAdModel 这层原始 nn.Module），它们的 forward 都直接接受
    (N, C, H, W) 的原始张量并在 eval 模式下返回 InferenceBatch(pred_score, anomaly_map, ...)，
    和 preprocess() 产出的合成/真实图像张量正好对得上，不需要额外套 anomalib 的 Batch 容器。
    """
    adapter_key = "repo_pipeline"

    _MODEL_CLASS_MAP = {
        "m.patchcore": "Patchcore",
        "m.rd4ad": "ReverseDistillation",
        "m.efficientad": "EfficientAd",
        "m.supersimplenet": "Supersimplenet",
    }

    def _load_real(self):
        weights = self.spec.get("weights_path")
        if not weights or not Path(weights).exists():
            raise FileNotFoundError(f"权重缺失: {weights}")

        cls_name = self._MODEL_CLASS_MAP.get(self.model_id)
        if cls_name is None:
            raise KeyError(f"未知的 anomaly 模型 id '{self.model_id}'，无法映射到 anomalib 模型类")

        anomalib_models = self.rt.optional_import("anomalib.models")
        if anomalib_models is None:
            raise ImportError("anomalib 未安装")
        model_cls = getattr(anomalib_models, cls_name, None)
        if model_cls is None:
            raise AttributeError(f"anomalib.models 中找不到 '{cls_name}'")

        torch = self.rt.optional_import("torch")
        device = self.rt.resolve_device() if torch is not None else "cpu"
        # weights_only=False: PyTorch >=2.6 defaults torch.load to weights_only=True, which
        # rejects anomalib's own custom classes (e.g. PrecisionType) pickled into the checkpoint.
        # Safe here because these are checkpoints this project trained itself, not arbitrary
        # third-party files.
        lightning_model = model_cls.load_from_checkpoint(weights, map_location=device, weights_only=False)
        lightning_model.eval()
        if torch is not None:
            lightning_model.to(device)
        return lightning_model

    def _infer_real(self, x) -> ModelOutput:
        torch = self.rt.optional_import("torch")
        with torch.no_grad():
            out = self.model.model(x)
        return ModelOutput(raw=out, backend="real", meta={"task": self.task})


class MoECLIPAdapter(BaseModelAdapter):
    """MoECLIP —— zero-shot 异常检测：加载 CLIP 骨干 + MoE adapter 头做真实推理。

    需要 models/MoECLIP 在 PYTHONPATH 上（本适配器按相对路径动态插入），
    以及两份权重：
      - clip_weights 指向的基础 CLIP 骨干（如 ViT-L-14-336px.pt，只读、不随本项目训练变化）
      - spec['weights_path'] 指向的 MoE 头微调权重（moe_last.pth）
    两者任一缺失都会在 _load_real() 里明确抛出，而不是静默用错误的权重。
    """
    adapter_key = "import_model_pkg"

    # models/MoECLIP 相对本文件的路径：benchmark_suite/common/models.py -> ../../models/MoECLIP
    _MOECLIP_ROOT = Path(__file__).resolve().parent.parent.parent / "models" / "MoECLIP"

    # 与 models/MoECLIP/test.py 的 argparser 默认值保持一致（该仓库唯一支持的 CLIP 骨干，
    # 硬编码在 model/clip.py 的 _MODEL_CKPT_PATHS 里，键名固定为 "ViT-L-14-336"，
    # 和 registry.yaml extra.clip_weights 里的文件名 "ViT-L-14-336px"（.pt 文件名）不是一回事）
    _CLIP_MODEL_NAME = "ViT-L-14-336"
    _DEFAULTS = dict(
        img_size=518, relu=False, use_paa=True, seg_proj_sharing_strategy="shared",
        image_adapt_weight=0.1, moe_r=8, moe_lora_alpha=16, moe_layers=(5, 11, 17, 23),
        use_fofs=True,
    )

    def _load_real(self):
        moe_head_path = self.spec.get("weights_path")
        if not moe_head_path or not Path(moe_head_path).exists():
            raise FileNotFoundError(f"MoE 头权重缺失: {moe_head_path}")

        clip_base_path = self._MOECLIP_ROOT / "model" / "ViT-L-14-336px.pt"
        if not clip_base_path.exists():
            raise FileNotFoundError(
                f"CLIP 基础骨干权重缺失: {clip_base_path}（只读权重，需按 MoECLIP README 手动下载）"
            )

        torch = self.rt.optional_import("torch")
        if torch is None:
            raise ImportError("torch 未安装")

        import sys
        if str(self._MOECLIP_ROOT) not in sys.path:
            sys.path.insert(0, str(self._MOECLIP_ROOT))

        clip_mod = self.rt.optional_import("model.clip")
        moe_adapter_mod = self.rt.optional_import("model.moe_adapter")
        if clip_mod is None or moe_adapter_mod is None:
            raise ImportError(f"无法从 {self._MOECLIP_ROOT} 导入 model.clip / model.moe_adapter（检查 MoECLIP 子模块是否就位）")

        extra = self.spec.get("extra", {})
        d = self._DEFAULTS
        device = self.rt.resolve_device()

        # 1. 基础 CLIP 骨干：走仓库自己的 create_model(pretrained="openai")，
        #    它内部会用 _MODEL_CKPT_PATHS[_CLIP_MODEL_NAME] 去读本地那份 .pt。
        clip_model = clip_mod.create_model(
            model_name=self._CLIP_MODEL_NAME,
            img_size=d["img_size"],
            device=device,
            pretrained="openai",
            require_pretrained=True,
        )
        clip_model.eval()

        # 2. 包上 MoE adapter 头
        moe_clip = moe_adapter_mod.MoECLIP(
            clip_model=clip_model,
            use_paa=d["use_paa"],
            seg_proj_sharing_strategy=d["seg_proj_sharing_strategy"],
            image_adapt_weight=d["image_adapt_weight"],
            moe_r=d["moe_r"],
            moe_lora_alpha=d["moe_lora_alpha"],
            moe_num_experts=extra.get("experts", 4),
            moe_top_k=extra.get("topk", 2),
            moe_layers=list(d["moe_layers"]),
            use_fofs=d["use_fofs"],
            relu=d["relu"],
        ).to(device)
        moe_clip.eval()

        # 3. MoE 头权重：checkpoint 是 {"text_adapter":..., "image_adapter":..., "epoch":...}
        #    这几个子模块分开存的字典，不是整模型一份 state_dict（和 test.py 载入逻辑保持一致）。
        checkpoint = torch.load(moe_head_path, map_location=device, weights_only=True)
        if "text_adapter" in checkpoint:
            moe_clip.text_adapter.load_state_dict(checkpoint["text_adapter"])
        if "image_adapter" in checkpoint:
            moe_clip.image_adapter.load_state_dict(checkpoint["image_adapter"])
        return moe_clip

    def _infer_real(self, x) -> ModelOutput:
        # MoECLIP.forward(x) 返回 (patch_features, det_feature, ...)（真实前向，见
        # models/MoECLIP/model/moe_adapter.py 和 test.py 的调用方式）。把这些原始特征转成
        # 最终的零样本异常分数，还需要 test.py 里另外做的文本 prompt 编码 + 相似度计算这一层，
        # 这属于具体测试单元（如 C5/M2）按需自行组装的评分逻辑，不是通用 adapter 该做的事——
        # 这里只保证“真实网络前向”这件事本身是真的，不是 mock。
        torch = self.rt.optional_import("torch")
        with torch.no_grad():
            out = self.model(x)
        return ModelOutput(raw=out, backend="real", meta={"task": self.task})


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
