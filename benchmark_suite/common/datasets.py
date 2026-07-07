"""统一数据接口 —— 把 5 个数据集抽象成同一 Sample 流（Handbook §2.3 / §2.4）。

服务于多个测试单元：
  - M2 量化-微缺陷抹除：需像素掩膜 → defect_size_bins() 按缺陷尺寸分层
  - C5 跨域漂移：需按域划分 → split_by_domain()
  - C2/C3：取图做前向

执行模型（§2.8）：
  - 本地 feasibility 或 data_mode=synthetic → 生成合成样本，不需真实文件，走通装配链路
  - server + data_mode=cloud/local → 从 root 扫描真实文件（yolo/mask/coco 布局）

真实布局因数据集而异，这里提供通用扫描（glob 图像 + 按命名配对掩膜/标签）；
个别数据集的特殊布局可后续在 spec 里加 layout 提示细化，不影响接口。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

try:
    import numpy as _np
except Exception:
    _np = None

from .runtime import Runtime

# 缺陷尺寸分层默认分箱（像素面积/最长边），供 M2 使用
DEFAULT_SIZE_BINS: List[Tuple[str, int, int]] = [
    ("micro", 0, 10),      # <10px：微缺陷（量化最易抹除）
    ("small", 10, 30),
    ("large", 30, 10**9),
]
_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class Sample:
    """统一样本容器。"""
    id: str
    image_path: Optional[str] = None
    array: Optional[Any] = None          # HxWx3 float32[0,1]（合成或已加载）
    label: str = "unknown"               # normal | anomaly | detection
    boxes: Optional[List] = None         # YOLO/COCO 框（可选）
    mask: Optional[Any] = None           # HxW uint8 {0,1}（若 has_mask）
    domain: Optional[str] = None         # 域标识（跨域划分用）
    meta: Dict[str, Any] = field(default_factory=dict)

    def load_array(self):
        if self.array is not None:
            return self.array
        if self.image_path and _np is not None:
            try:
                from PIL import Image
                return _np.asarray(Image.open(self.image_path).convert("RGB"),
                                   dtype="float32") / 255.0
            except Exception:
                return None
        return None


class BaseDataset:
    def __init__(self, dataset_id: str, spec: Dict[str, Any], runtime: Runtime,
                 root: Optional[str] = None, split: str = "test",
                 n_synthetic: int = 8, normal_only: bool = False):
        self.id = dataset_id
        self.spec = spec
        self.rt = runtime
        self.root = Path(root) if root else None
        self.split = split
        self.n_synthetic = n_synthetic
        # normal_only：只产出正常样本（异常检测训练用）。当数据集 train_normal_only 或
        # normal_subset_available 时可切出"仅正常"子集。
        self.normal_only = normal_only
        self.fmt = spec.get("format")
        self.has_mask = bool(spec.get("has_mask"))
        self.train_normal_only = bool(spec.get("train_normal_only"))
        self.normal_subset_available = bool(spec.get("normal_subset_available"))
        self.domain_field = spec.get("domain_field")
        self._use_synthetic = (
            self.rt.data_mode == "synthetic" or self.root is None or not self.root.exists()
        )

    # ------------------------------------------------------------------ #
    def __iter__(self) -> Iterator[Sample]:
        src = self._iter_synthetic() if self._use_synthetic else self._iter_real()
        for s in src:
            if self.normal_only and s.label == "anomaly":
                continue                      # 切出"仅正常"子集
            yield s

    def can_train_anomaly(self) -> bool:
        """异常检测能否在此数据集训练：原生仅正常，或可切正常子集。"""
        return self.train_normal_only or self.normal_subset_available

    def __len__(self) -> int:
        return self.n_synthetic if self._use_synthetic else sum(1 for _ in self._iter_real())

    def domains(self) -> List[str]:
        if not self.domain_field:
            return []
        if self._use_synthetic:
            return [f"{self.domain_field}_{i}" for i in range(3)]
        # 真实：以 root 下一层子目录名作为域
        return sorted({p.name for p in (self.root or Path()).glob("*") if p.is_dir()})

    # ------------------------------------------------------------------ #
    def _iter_synthetic(self) -> Iterator[Sample]:
        assert _np is not None, "合成样本需要 numpy"
        res = 64  # 合成用小图，够走通链路即可
        doms = self.domains() or [None]
        for i in range(self.n_synthetic):
            arr = _np.random.rand(res, res, 3).astype("float32")
            is_anom = (i % 2 == 1) and not (self.train_normal_only and self.split == "train")
            mask = None
            if self.has_mask:
                mask = _np.zeros((res, res), dtype="uint8")
                if is_anom:  # 放一个缺陷方块，尺寸跨 micro/small/large 三档
                    s = int(_np.random.randint(4, min(res - 2, 52)))
                    y, x = _np.random.randint(0, res - s, 2)
                    mask[y:y+s, x:x+s] = 1
            yield Sample(
                id=f"{self.id}:syn{i}", array=arr,
                label=("anomaly" if is_anom else "normal") if self.fmt != "yolo" else "detection",
                boxes=[[0, 0.5, 0.5, 0.2, 0.2]] if (self.fmt == "yolo" and is_anom) else None,
                mask=mask, domain=doms[i % len(doms)],
                meta={"synthetic": True},
            )

    def _iter_real(self) -> Iterator[Sample]:
        # 通用扫描：递归找图像；尝试按命名找并行掩膜（*_mask/ground_truth）
        for p in sorted(self.root.rglob("*")):
            if p.suffix.lower() not in _IMG_EXT:
                continue
            if "mask" in p.stem.lower() or "ground_truth" in str(p).lower():
                continue  # 跳过掩膜本身
            mask_path = self._guess_mask(p) if self.has_mask else None
            domain = p.parent.name if self.domain_field else None
            label = "normal" if "good" in str(p).lower() or "normal" in str(p).lower() else "anomaly"
            yield Sample(
                id=f"{self.id}:{p.stem}", image_path=str(p),
                label=label if self.fmt != "yolo" else "detection",
                mask=str(mask_path) if mask_path else None,
                domain=domain, meta={"path": str(p)},
            )

    def _guess_mask(self, img: Path) -> Optional[Path]:
        cands = [
            img.with_name(img.stem + "_mask" + img.suffix),
            img.with_name(img.stem + "_mask.png"),
        ]
        gt = str(img).replace("/test/", "/ground_truth/")
        cands.append(Path(gt).with_name(img.stem + "_mask.png"))
        for c in cands:
            if c.exists():
                return c
        return None


# --------------------------------------------------------------------------- #
# 缺陷尺寸分层（M2 核心）
# --------------------------------------------------------------------------- #
def defect_size_bins(mask, bins=DEFAULT_SIZE_BINS) -> Dict[str, int]:
    """统计掩膜中各连通缺陷按尺寸落入的分箱计数（尺寸=连通域最长边像素）。

    优先用 scipy.ndimage 做连通域标注；无 scipy 时退化为"整块掩膜算一个缺陷"。
    """
    if _np is None:
        return {b[0]: 0 for b in bins}
    m = _np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 0]
    m = (m > 0).astype("uint8")
    counts = {b[0]: 0 for b in bins}
    if m.sum() == 0:
        return counts

    comps = []
    try:
        from scipy import ndimage
        lab, n = ndimage.label(m)
        for i in range(1, n + 1):
            ys, xs = _np.where(lab == i)
            comps.append(max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1))
    except Exception:
        ys, xs = _np.where(m > 0)
        comps.append(max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1))

    for size in comps:
        for name, lo, hi in bins:
            if lo <= size < hi:
                counts[name] += 1
                break
    return counts


# --------------------------------------------------------------------------- #
# 跨域划分（C5 核心）
# --------------------------------------------------------------------------- #
def split_by_domain(dataset: BaseDataset, train_domains: List[str],
                    test_domains: List[str]) -> Dict[str, List[Sample]]:
    """按域把样本划成 train/test（真实的跨工位/跨类目分布，而非增广）。"""
    buckets = {"train": [], "test": []}
    for s in dataset:
        if s.domain in train_domains:
            buckets["train"].append(s)
        elif s.domain in test_domains:
            buckets["test"].append(s)
    return buckets


# --------------------------------------------------------------------------- #
def get_dataset(dataset_id: str, registry, runtime: Runtime,
                root: Optional[str] = None, split: str = "test",
                n_synthetic: int = 8, normal_only: bool = False) -> BaseDataset:
    spec = registry.datasets[dataset_id]
    return BaseDataset(dataset_id, spec, runtime, root=root, split=split,
                       n_synthetic=n_synthetic, normal_only=normal_only)


def feasibility_check(dataset_id: str, registry, runtime: Runtime) -> Dict[str, Any]:
    """数据接口可行性：取一个样本，验证字段；若 has_mask，验证尺寸分层可算。"""
    out = {"dataset": dataset_id, "ok": False, "mode": "synthetic", "note": ""}
    try:
        ds = get_dataset(dataset_id, registry, runtime)
        out["mode"] = "synthetic" if ds._use_synthetic else "real"
        s = next(iter(ds))
        checks = [s.id, s.label]
        note = f"样本 ok label={s.label} domain={s.domain}"
        if ds.has_mask and s.mask is not None and not isinstance(s.mask, str):
            bins = defect_size_bins(s.mask)
            note += f" size_bins={bins}"
        out["ok"] = all(c is not None for c in checks)
        out["note"] = note
    except Exception as e:
        out["note"] = f"{type(e).__name__}: {e}"
    return out
