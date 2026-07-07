"""统一结果契约 result.json 的构建与落盘（Handbook §2.6）。

每个测试单元运行后调用 build_result(...) 得到标准字典，再 write_result(...) 落盘到
results/<test_id>/<timestamp>/result.json。字段名即代码 key，供 report.py 聚合。
"""
from __future__ import annotations
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .env import collect_env

_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

_TOP_FIELDS = [
    "test_id", "test_name", "timestamp", "measurement_type", "hardware_id",
    "env", "config", "metrics", "hypothesis", "artifacts", "notes",
]
_VERDICTS = {"unfilled", "supported", "partial", "not_supported"}


def build_result(test_id: str, test_name: str, *, measurement_type: str,
                 hardware_id: str, config: Dict[str, Any],
                 metrics: Dict[str, Any] | None = None,
                 hypothesis_id: str | None = None,
                 verdict: str = "unfilled", evidence: str = "",
                 artifacts: List[str] | None = None, notes: str = "",
                 env: Dict[str, Any] | None = None) -> Dict[str, Any]:
    assert measurement_type in ("real", "proxy"), measurement_type
    assert verdict in _VERDICTS, verdict
    return {
        "test_id": test_id,
        "test_name": test_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "measurement_type": measurement_type,
        "hardware_id": hardware_id,
        "env": env if env is not None else collect_env(),
        "config": config,
        "metrics": metrics or {},
        "hypothesis": {"id": hypothesis_id, "verdict": verdict, "evidence": evidence},
        "artifacts": artifacts or [],
        "notes": notes,
    }


def validate_result(res: Dict[str, Any]) -> List[str]:
    issues = [f"缺字段 '{f}'" for f in _TOP_FIELDS if f not in res]
    if res.get("measurement_type") not in ("real", "proxy"):
        issues.append("measurement_type 必须是 real|proxy")
    if res.get("hypothesis", {}).get("verdict") not in _VERDICTS:
        issues.append("hypothesis.verdict 非法")
    return issues


def write_result(res: Dict[str, Any], base: str | Path | None = None) -> Path:
    issues = validate_result(res)
    if issues:
        raise ValueError("result 不合规: " + "; ".join(issues))
    base = Path(base) if base else _RESULTS_DIR
    ts = re.sub(r"[:.]", "-", res["timestamp"])
    out_dir = base / res["test_id"] / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "result.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
