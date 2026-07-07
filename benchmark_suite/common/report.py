"""报告层 —— 聚合各测试单元结论为「逐假设判定」总表（Handbook §5）。

两种数据来源：
  - 内存：运行全部已实现测试单元（当前 --mode），收集 result 字典（不落盘）。适合 local 演示。
  - 读盘：扫描 results/<test_id>/<timestamp>/result.json，取每个 test 最新一条。适合 server。

产出：
  - 控制台总表（按维度分组 + 判定计数）
  - Markdown 报告 results/report.md

判定不预判：verdict 来自各单元实测/演示；local 模式下为合成演示，报告会显式标注。
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .registry import load_registry, load_yaml

_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
_DIM_LABEL = {"compute": "计算", "memory": "内存", "comm": "通信"}
_VERDICT_LABEL = {"supported": "支持", "partial": "部分支持",
                  "not_supported": "不支持", "unfilled": "待填", "error": "未运行/出错"}


# --------------------------------------------------------------------------- #
# 采集
# --------------------------------------------------------------------------- #
def collect_from_dir(base: Optional[Path] = None) -> Dict[str, dict]:
    """从 results/ 读取每个 test_id 最新一条 result.json。"""
    base = Path(base) if base else _RESULTS_DIR
    latest: Dict[str, dict] = {}
    if not base.exists():
        return latest
    for f in base.rglob("result.json"):
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        tid = r.get("test_id")
        if tid and (tid not in latest or r.get("timestamp", "") > latest[tid].get("timestamp", "")):
            latest[tid] = r
    return latest


def collect_in_memory(dispatch: Dict[str, str], registry, runtime) -> Dict[str, dict]:
    """在当前 runtime 下运行全部已实现单元，收集 result（不落盘）。出错的单元记为 error。"""
    import importlib
    out: Dict[str, dict] = {}
    for tid, target in dispatch.items():
        mod_name, fn_name = target.split(":")
        try:
            fn = getattr(importlib.import_module(mod_name), fn_name)
            out[tid] = fn(registry=registry, runtime=runtime, write=False)
        except Exception as e:
            out[tid] = {"test_id": tid, "hypothesis": {"id": None, "verdict": "error",
                        "evidence": f"{type(e).__name__}: {e}"},
                        "measurement_type": "-", "config": {"mode": getattr(runtime, "name", "-")}}
    return out


# --------------------------------------------------------------------------- #
# 汇总
# --------------------------------------------------------------------------- #
def build_summary(results: Dict[str, dict], registry, hypotheses: dict) -> List[dict]:
    rows = []
    for tid, t in registry.test_units.items():        # 按注册表顺序 C1..N3
        r = results.get(tid)
        hyp_id = t.get("hypothesis")
        hyp = hypotheses.get(hyp_id, {}) if hyp_id else {}
        h = (r or {}).get("hypothesis", {})
        rows.append({
            "test_id": tid,
            "dim": t.get("dim"),
            "dim_label": _DIM_LABEL.get(t.get("dim"), t.get("dim")),
            "name": t.get("name"),
            "hyp_id": hyp_id,
            "hyp_type": hyp.get("type", "-"),
            "statement": hyp.get("statement", ""),
            "verdict": h.get("verdict", "未运行") if r else "未运行",
            "evidence": h.get("evidence", ""),
            "measurement_type": (r or {}).get("measurement_type", "-"),
        })
    return rows


def verdict_counts(rows: List[dict]) -> Dict[str, int]:
    c: Dict[str, int] = {}
    for row in rows:
        c[row["verdict"]] = c.get(row["verdict"], 0) + 1
    return c


# --------------------------------------------------------------------------- #
# 渲染
# --------------------------------------------------------------------------- #
def render_console(rows: List[dict], mode: str) -> None:
    print(f"\n开销挑战验证 —— 逐假设判定总表（mode={mode}）\n")
    print(f"{'测试':4s} {'维度':4s} {'假设':4s} {'类型':6s} {'判定':10s} 证据")
    last_dim = None
    for r in rows:
        if r["dim"] != last_dim:
            print(f"— {r['dim_label']}开销 —")
            last_dim = r["dim"]
        v = _VERDICT_LABEL.get(r["verdict"], r["verdict"])
        print(f"  {r['test_id']:3s} {r['dim_label']:4s} {str(r['hyp_id'] or '-'):4s} "
              f"{r['hyp_type']:6s} {v:10s} {r['evidence'][:46]}")
    c = verdict_counts(rows)
    summary = "  ".join(f"{_VERDICT_LABEL.get(k, k)}={v}" for k, v in c.items())
    print(f"\n计数: {summary}")


def render_markdown(rows: List[dict], mode: str, measurement_note: str = "") -> str:
    from datetime import datetime, timezone
    lines = [
        "# 开销挑战验证 —— 逐假设判定报告",
        "",
        f"> 生成时间：{datetime.now(timezone.utc).isoformat()}　|　执行档：`{mode}`",
        "",
    ]
    if measurement_note:
        lines += [f"> {measurement_note}", ""]
    # 计数
    c = verdict_counts(rows)
    lines.append("**判定计数**：" + "，".join(f"{_VERDICT_LABEL.get(k, k)} {v}" for k, v in c.items()))
    lines.append("")
    # 按维度分组表
    for dim in ["compute", "memory", "comm"]:
        drows = [r for r in rows if r["dim"] == dim]
        if not drows:
            continue
        lines.append(f"## {_DIM_LABEL[dim]}开销")
        lines.append("")
        lines.append("| 测试 | 假设 | 类型 | 挑战陈述 | 判定 | 证据 |")
        lines.append("|---|---|---|---|---|---|")
        for r in drows:
            v = _VERDICT_LABEL.get(r["verdict"], r["verdict"])
            lines.append(f"| {r['test_id']} | {r['hyp_id'] or '-'} | {r['hyp_type']} | "
                         f"{r['statement']} | **{v}** | {r['evidence']} |")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
def generate(dispatch: Dict[str, str], runtime, source: str = "memory",
             out_path: Optional[Path] = None) -> Dict[str, Any]:
    """主入口：采集 → 汇总 → 渲染（控制台 + 写 Markdown）。返回 summary。"""
    registry = load_registry()
    hypotheses = load_yaml("hypotheses.yaml")
    if source == "disk":
        results = collect_from_dir()
        note = "数据来源：results/ 落盘结果。"
    else:
        results = collect_in_memory(dispatch, registry, runtime)
        note = ("数据来源：内存运行全部单元。**当前为 local 可行性模式，判定来自合成演示，非真实测量。**"
                if getattr(runtime, "is_feasibility", False) else "数据来源：内存运行全部单元。")
    rows = build_summary(results, registry, hypotheses)
    mode = getattr(runtime, "name", "-")
    render_console(rows, mode)
    md = render_markdown(rows, mode, note)
    out = Path(out_path) if out_path else _RESULTS_DIR / "report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"\n✔ Markdown 报告已写入: {out}")
    return {"rows": rows, "counts": verdict_counts(rows), "report_path": str(out)}
