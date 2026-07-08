#!/usr/bin/env python3
"""编排入口（阶段0 骨架）—— Verification Handbook v3 §2.7 / §4。

当前已实现：注册表加载/校验、环境采集、结果契约自测、前置检查与权重准备。

用法:
    python run.py --env              # 打印环境元数据
    python run.py --validate         # 校验注册表一致性
    python run.py --list             # 列出模型/数据集/测试单元
    python run.py --demo-result      # 走通 result.json 契约（写一条占位结果）
    python run.py --status           # 显示各测试单元的实现状态
    python run.py --prepare          # 执行前置检查与权重准备 (默认 local 模式)
    python run.py --prepare --mode server # 真实测量前的权重准备（若有可训练模型缺失，会拉起训练）
    python run.py --check            # 本地可行性验证：各模型 load→合成输入→infer（mock 回退）
    python run.py --check --mode server   # 服务器真实模式（缺依赖会报错，不 mock）

执行模型（§2.8）：--mode local（默认，仅可行性验证）| server（CUDA GPU 真实测量，云端数据）。
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.registry import load_registry, load_yaml  # noqa: E402
from common.env import collect_env                      # noqa: E402

# 测试单元实现状态：逐个填入
IMPLEMENTED = {"C1", "C2", "C3", "C4", "C5", "C6", "M1", "M2", "N1", "N2", "N3"}

# 测试单元分发表：test_id -> "module:function"
TEST_DISPATCH = {
    "C1": "compute.c1_power_thermal:run",
    "C2": "compute.c2_resolution_sweep:run",
    "C3": "compute.c3_concurrency:run",
    "C4": "compute.c4_changeover_training:run",
    "C5": "compute.c5_domain_shift:run",
    "C6": "compute.c6_upgrade_consistency:run",
    "M1": "memory.m1_bandwidth:run",
    "M2": "memory.m2_quantization:run",
    "N1": "communication.n1_model_sync:run",
    "N2": "communication.n2_event_callback:run",
    "N3": "communication.n3_hard_realtime:run",
}


def cmd_env():
    print(json.dumps(collect_env(), indent=2, ensure_ascii=False))


def cmd_validate():
    reg = load_registry()
    issues = reg.validate()
    if not issues:
        print("✔ 注册表校验通过：hardware/models/datasets/test_units 引用一致。")
        return 0
    print("[X] 注册表存在问题：")
    for i in issues:
        print("  -", i)
    return 1


def cmd_list():
    reg = load_registry()
    print("模型:")
    for mid, m in reg.models.items():
        print(f"  {mid:14s} {m['name']:12s} paradigm={m['paradigm']:14s} repo={m.get('source_repo')}")
    print("\n数据集:")
    for did, d in reg.datasets.items():
        print(f"  {did:12s} {d['name']:12s} mask={str(d.get('has_mask')):5s} format={d.get('format')}")
    print("\n测试单元:")
    for tid, t in reg.test_units.items():
        print(f"  {tid:3s} [{t['dim']:7s}] {t['name']:16s} -> {reg.hypothesis_of(tid)}")


def cmd_status():
    reg = load_registry()
    print(f"{'测试':4s} {'维度':8s} {'假设':5s} {'状态'}")
    for tid, t in reg.test_units.items():
        state = "implemented" if tid in IMPLEMENTED else "pending"
        print(f"{tid:4s} {t['dim']:8s} {reg.hypothesis_of(tid) or '-':5s} {state}")
    print(f"\n已实现 {len(IMPLEMENTED)}/{len(reg.test_units)} 个测试单元（阶段0 为脚手架，测试单元待后续步骤）。")


def cmd_check(mode: str):
    from common.runtime import load_runtime
    from common.models import feasibility_check
    reg = load_registry()
    rt = load_runtime(mode)
    try:
        dev = rt.resolve_device()
    except RuntimeError as e:
        print(f"[X] 无法在当前环境以 '{mode}' 模式运行: {e}")
        print("    提示: server 模式需在装有 CUDA+torch 的 GPU 服务器上运行；本地请用 --mode local。")
        return 2
    print(f"运行档: {rt.name}  role={rt.role}  device(req)={rt.device}->{dev}  "
          f"allow_mock={rt.allow_mock}  data={rt.data_mode}")
    print(f"循环参数: warmup={rt.warmup} iters={rt.iters} repeats={rt.repeats}\n")
    print(f"[模型适配层] {'模型':14s} {'后端':6s} {'通过':4s} 说明")
    all_ok = True
    for mid in reg.models:
        r = feasibility_check(mid, reg, rt)
        ok = "OK" if r["ok"] else "FAIL"
        all_ok = all_ok and r["ok"]
        print(f"  {mid:12s} {str(r['backend'] or '-'):6s} {ok:4s} {r['note']}")

    from common.datasets import feasibility_check as ds_check
    print(f"\n[数据接口]   {'数据集':12s} {'来源':10s} {'通过':4s} 说明")
    for did in reg.datasets:
        r = ds_check(did, reg, rt)
        ok = "OK" if r["ok"] else "FAIL"
        all_ok = all_ok and r["ok"]
        print(f"  {did:12s} {r['mode']:10s} {ok:4s} {r['note']}")
    print()
    if rt.is_feasibility:
        print("说明: 本地为可行性验证，backend=mock 属正常（无 GPU/权重时占位走通装配链路）。")
    return 0 if all_ok else 1


def cmd_test(test_id: str, model_id: str, dataset_id: str, mode: str):
    import importlib
    from common.runtime import load_runtime
    if test_id not in TEST_DISPATCH:
        print(f"[X] 测试单元 '{test_id}' 尚未实现。已实现: {sorted(IMPLEMENTED)}")
        return 2
    rt = load_runtime(mode)
    mod_name, fn_name = TEST_DISPATCH[test_id].split(":")
    fn = getattr(importlib.import_module(mod_name), fn_name)
    reg = load_registry()
    # 兼容性守卫：仅当用户显式同时指定 model 与 dataset 时校验
    if model_id and dataset_id:
        from common import compat
        ok, reasons = compat.check(reg, test_id, model_id, dataset_id)
        if not ok:
            print(f"[X] 非法组合：{test_id} × {model_id} × {dataset_id}")
            for r in reasons:
                print("   -", r)
            print("   提示：用 `python run.py --compat --test", test_id, "` 查看合法组合。")
            return 3
    print(f"运行 {test_id}  model={model_id} dataset={dataset_id}  mode={rt.name}({rt.role})")
    res = fn(model_id, dataset_id, registry=reg, runtime=rt)
    h = res["hypothesis"]
    print(f"\n假设 {h['id']}: {h['verdict']}  —— {h['evidence']}")
    m = res["metrics"]

    def f2(v): return f"{v:.2f}" if isinstance(v, (int, float)) and v == v else "  - "

    # M2：分尺寸召回率表
    rbs = m.get("recall_by_size_by_precision", {})
    if rbs:
        print("\n各精度分尺寸召回率:")
        print(f"  {'精度':10s} {'micro':>7s} {'small':>7s} {'large':>7s}")
        for prec, rc in rbs.items():
            print(f"  {prec:10s} {f2(rc.get('micro')):>7s} {f2(rc.get('small')):>7s} {f2(rc.get('large')):>7s}")
    # C2：分组斜率表
    if "per_model" in m and "slope_by_group" in m:
        print("\n各模型 吞吐-分辨率 log-log 斜率（越负=随分辨率下降越快）:")
        print(f"  {'模型':12s} {'分组':16s} {'斜率':>7s}")
        for mid, d in m["per_model"].items():
            print(f"  {mid:12s} {d['group']:16s} {f2(d['slope']):>7s}")
        sg = m.get("slope_gap", {}).get("value")
        print(f"  slope_gap(卷积均值-注意力均值) = {f2(sg)}")
    # 通用：summary_lines（C1/C3/C4/C5/C6 等）
    if not rbs and not ("per_model" in m and "slope_by_group" in m):
        for line in m.get("summary_lines", []):
            print(f"  {line}")
    if res.get("_written_to"):
        print(f"\n已写入: {res['_written_to']}")
    else:
        print("\n(local 可行性：未写 result.json；契约校验已通过)")
    return 0


def cmd_compat(test_id: str | None):
    from common import compat
    reg = load_registry()
    tids = [test_id] if test_id else list(reg.test_units)
    for tid in tids:
        t = reg.test_units[tid]
        combos = compat.legal_combos(reg, tid)
        needs = t.get("needs", [])
        print(f"\n{tid} [{t['dim']}] {t['name']}  假设 {reg.hypothesis_of(tid)}"
              f"{'  needs=' + ','.join(needs) if needs else ''}")
        if not combos:
            print("  （无合法组合——检查数据集/模型约束）")
            continue
        by_model = {}
        for c in combos:
            by_model.setdefault(c["model"], []).append(c["dataset"])
        for mid, dss in by_model.items():
            task = reg.models[mid].get("task")
            print(f"  {mid:12s} ({task:16s}) → {', '.join(dss)}")
    return 0


def cmd_report(mode: str, source: str):
    from common.runtime import load_runtime
    from common.report import generate
    rt = load_runtime(mode)
    generate(TEST_DISPATCH, rt, source=source)
    return 0


def cmd_demo_result():
    from common.result import build_result, write_result
    reg = load_registry()
    res = build_result(
        test_id="C0", test_name="scaffold_selftest",
        measurement_type="real", hardware_id="hw.cloud",
        config={"note": "阶段0 自测：验证 result.json 契约可写"},
        metrics={"placeholder": True},
        hypothesis_id=None, verdict="unfilled",
        notes="这是脚手架自测产物，非真实测量。",
    )
    path = write_result(res)
    print(f"✔ 已写入示例结果: {path}")


def main():
    ap = argparse.ArgumentParser(description="开销挑战验证框架 —— 编排入口（阶段0）")
    ap.add_argument("--env", action="store_true")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--prepare", action="store_true", help="执行前置检查与权重准备")
    ap.add_argument("--demo-result", action="store_true")
    ap.add_argument("--check", action="store_true", help="模型适配层可行性验证")
    ap.add_argument("--mode", choices=["local", "server"], default="local",
                    help="执行档：local=可行性验证(默认) | server=真实测量")
    ap.add_argument("--test", metavar="ID", help="运行某测试单元，如 M2")
    ap.add_argument("--compat", action="store_true", help="显示每个测试的合法(模型×数据集)组合")
    ap.add_argument("--report", action="store_true", help="聚合同步全部单元 → 逐假设判定总表 + results/report.md")
    ap.add_argument("--source", choices=["memory", "disk"], default="memory",
                    help="报告数据来源：memory=内存跑全部单元(默认) | disk=读 results/ 落盘")
    ap.add_argument("--model", default=None, help="--test 用的模型 id（缺省时各测试用自身默认或跑分组对比）")
    ap.add_argument("--dataset", default="d.mvtec", help="--test 用的数据集 id")
    args = ap.parse_args()

    if args.env:
        cmd_env()
    elif args.validate:
        sys.exit(cmd_validate())
    elif args.list:
        cmd_list()
    elif args.status:
        cmd_status()
    elif args.prepare:
        from common.prepare import check_and_prepare
        sys.exit(check_and_prepare(args.mode))
    elif args.check:
        sys.exit(cmd_check(args.mode))
    elif args.compat:
        sys.exit(cmd_compat(args.test))
    elif args.test:
        sys.exit(cmd_test(args.test, args.model, args.dataset, args.mode))
    elif args.report:
        sys.exit(cmd_report(args.mode, args.source))
    elif args.demo_result:
        cmd_demo_result()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
