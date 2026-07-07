"""加载中央注册表并做兼容性校验（Handbook §2.2-2.4）。

不依赖 torch；只需 pyyaml。提供：
  load_registry(path) -> Registry
  Registry.validate() -> List[str]  (问题列表，空=通过)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


@dataclass
class Registry:
    hardware: Dict[str, Any] = field(default_factory=dict)
    models: Dict[str, Any] = field(default_factory=dict)
    datasets: Dict[str, Any] = field(default_factory=dict)
    paradigm_groups: Dict[str, List[str]] = field(default_factory=dict)
    test_units: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)

    # -- 查询辅助 --
    def models_by_paradigm(self, paradigm: str) -> List[str]:
        return [mid for mid, m in self.models.items() if m.get("paradigm") == paradigm]

    def datasets_with_mask(self) -> List[str]:
        return [did for did, d in self.datasets.items() if d.get("has_mask")]

    def hypothesis_of(self, test_id: str) -> str | None:
        return self.test_units.get(test_id, {}).get("hypothesis")

    # -- 校验 --
    def validate(self) -> List[str]:
        issues: List[str] = []

        # 1) 硬件 proxy_by 引用有效
        for hid, h in self.hardware.items():
            pb = h.get("proxy_by")
            if pb and pb not in self.hardware:
                issues.append(f"[hardware:{hid}] proxy_by='{pb}' 不在硬件注册表")

        # 2) 范式分组引用有效模型
        for g, ids in self.paradigm_groups.items():
            for mid in ids:
                if mid not in self.models:
                    issues.append(f"[paradigm_groups:{g}] 未知模型 '{mid}'")

        # 3) 测试单元引用有效
        SPECIAL_MODELS = {"all", "any"}
        SPECIAL_DS = {"any"}
        for tid, t in self.test_units.items():
            for did in t.get("datasets", []):
                if did not in SPECIAL_DS and did not in self.datasets:
                    issues.append(f"[test:{tid}] 未知数据集 '{did}'")
            mval = t.get("models")
            if isinstance(mval, str):
                mlist = [] if mval in SPECIAL_MODELS else [mval]
            else:
                mlist = mval or []
            for mid in mlist:
                if mid not in self.models:
                    issues.append(f"[test:{tid}] 未知模型 '{mid}'")
            # 4) 语义约束：需要像素掩膜的测试，其数据集必须 has_mask
            if "pixel_mask" in t.get("needs", []):
                for did in t.get("datasets", []):
                    if did in self.datasets and not self.datasets[did].get("has_mask"):
                        issues.append(f"[test:{tid}] needs=pixel_mask 但 '{did}' 无掩膜")

        return issues


def load_registry(path: str | Path | None = None) -> Registry:
    p = Path(path) if path else _CONFIG_DIR / "registry.yaml"
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return Registry(
        hardware=data.get("hardware", {}),
        models=data.get("models", {}),
        datasets=data.get("datasets", {}),
        paradigm_groups=data.get("paradigm_groups", {}),
        test_units=data.get("test_units", {}),
        raw=data,
    )


def load_yaml(name: str) -> dict:
    """加载 configs/ 下任意 yaml（measurement / hypotheses）。"""
    return yaml.safe_load((_CONFIG_DIR / name).read_text(encoding="utf-8"))
