"""前置检查与权重准备模块。

负责：
1. 采用以“模型”为中心的统一校验流，清晰展示每个模型的基础权重（Backbone）与训练权重（Trained Weights）的状态。
2. 校验完成后，提供交互式菜单（1..5），允许用户分批选择并触发训练/获取流程。
3. 安全约束：
   - 在本地模式（local）下，如果用户试图选择任何需要训练或抓取的项，系统会直接否决并中止，防止超负荷计算。
   - 在 MoECLIP 基础权重缺失时，安全拦截其 MoE 头的训练。
"""
from __future__ import annotations
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, Any, List
from .registry import load_registry
from .runtime import Runtime, load_runtime

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent


def check_and_prepare(mode: str) -> int:
    reg = load_registry()
    rt = load_runtime(mode)
    
    print(f"=== 开始前置检查与权重准备 (运行档: {rt.name}) ===\n")
    
    # 1. 检测各权重文件的存在状态
    yolo_status = "✔ 已就位"
    missing_yolos = []
    for yolo_id in ["m.yolov8n", "m.yolov8s", "m.yolov11n"]:
        wpath = reg.models[yolo_id].get("weights_path")
        abs_wpath = WORKSPACE_ROOT / wpath if wpath else None
        if not abs_wpath or not abs_wpath.exists():
            missing_yolos.append(yolo_id)
    if missing_yolos:
        yolo_status = f"✘ 缺失 (缺少: {', '.join(missing_yolos)})"
        
    clip_base_path = WORKSPACE_ROOT / "models/MoECLIP/model/ViT-L-14-336px.pt"
    moe_head_path = WORKSPACE_ROOT / reg.models["m.moeclip"].get("weights_path")
    clip_base_ok = clip_base_path.exists()
    moe_head_ok = moe_head_path.exists()
    
    patchcore_path = WORKSPACE_ROOT / reg.models["m.patchcore"].get("weights_path")
    rd4ad_path = WORKSPACE_ROOT / reg.models["m.rd4ad"].get("weights_path")
    efficientad_path = WORKSPACE_ROOT / reg.models["m.efficientad"].get("weights_path")
    
    patchcore_ok = patchcore_path.exists()
    rd4ad_ok = rd4ad_path.exists()
    efficientad_ok = efficientad_path.exists()

    # 打印状态报告
    print("各模型权重状态报告：")
    
    # YOLO
    print("[m.yolov8n/v8s/v11n] (YOLO 系列):")
    if len(missing_yolos) == 0:
        print(f"  ✔ 预训练权重已就位")
    else:
        print(f"  ✘ 预训练权重缺失 (缺少: {', '.join(missing_yolos)})")
        print(f"    -> 指引: 请使用 fabric-defect-detection 仓库中获取权重-U 的脚本进行下载。")
        
    # MoECLIP
    print("[m.moeclip] (MoECLIP):")
    if clip_base_ok:
        print(f"  ✔ 基础骨干权重已就位: models/MoECLIP/model/ViT-L-14-336px.pt")
    else:
        print(f"  ✘ 基础骨干权重缺失: models/MoECLIP/model/ViT-L-14-336px.pt")
        print(f"    -> 指引: 请按照 models/MoECLIP/README.md 下载并放入该目录。")
    if moe_head_ok:
        print(f"  ✔ MoE头微调权重已就位: models/MoECLIP/ckpt/baseline/moe_last.pth")
    else:
        print(f"  ✘ MoE头微调权重缺失: models/MoECLIP/ckpt/baseline/moe_last.pth")
        
    # PatchCore
    print("[m.patchcore] (PatchCore):")
    print(f"  {'✔' if patchcore_ok else '✘'} 训练权重{'已就位' if patchcore_ok else '缺失'}: {reg.models['m.patchcore'].get('weights_path')}")
    
    # RD4AD
    print("[m.rd4ad] (RD4AD):")
    print(f"  {'✔' if rd4ad_ok else '✘'} 训练权重{'已就位' if rd4ad_ok else '缺失'}: {reg.models['m.rd4ad'].get('weights_path')}")
    
    # EfficientAD
    print("[m.efficientad] (EfficientAD):")
    print(f"  {'✔' if efficientad_ok else '✘'} 训练权重{'已就位' if efficientad_ok else '缺失'}: {reg.models['m.efficientad'].get('weights_path')}")
    print()

    # 2. 交互式分批菜单
    menu_options = {
        "1": {"name": "YOLO 系列预训练权重", "status": yolo_status, "type": "pretrained"},
        "2": {"name": "PatchCore 异常检测模型", "status": "✔ 已就位" if patchcore_ok else "✘ 缺失", "type": "trainable", "mid": "m.patchcore"},
        "3": {"name": "RD4AD 异常检测模型", "status": "✔ 已就位" if rd4ad_ok else "✘ 缺失", "type": "trainable", "mid": "m.rd4ad"},
        "4": {"name": "EfficientAD 异常检测模型", "status": "✔ 已就位" if efficientad_ok else "✘ 缺失", "type": "trainable", "mid": "m.efficientad"},
        "5": {"name": "MoECLIP 零样本异常检测模型 (MoE头)", "status": "✔ 已就位" if moe_head_ok else "✘ 缺失", "type": "trainable", "mid": "m.moeclip"}
    }

    while True:
        print("请选择要处理的权重项目（输入序号，支持逗号分隔如 2,3；输入 q 退出）：")
        for k, opt in menu_options.items():
            print(f"  [{k}] {opt['name']} -> 状态: [{opt['status']}]")
        
        try:
            choice = input("你的选择: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\n[INFO] 准备流程已中止。")
            break
            
        if choice in ["q", "quit", "exit", ""]:
            print("[INFO] 退出准备流程。")
            break
            
        # 解析选择的项目
        parts = [p.strip() for p in choice.split(",") if p.strip()]
        invalid_choices = [p for p in parts if p not in menu_options]
        if invalid_choices:
            print(f"[X] 无效序号: {', '.join(invalid_choices)}，请重新选择。\n")
            continue
            
        # 本地模式（local）下，如果用户做出选择，直接予以否决
        if rt.is_feasibility:
            print(f"\n[拒绝执行] 当前运行在本地模式 (local)，不允许在本地启动模型训练或获取权重，这会超出本地计算量！")
            print(f"请在 GPU 服务器上使用 --mode server 运行该准备程序。\n")
            continue

        # 执行选定项目 (Server 模式)
        for part in parts:
            opt = menu_options[part]
            print(f"\n>>> 开始处理: [{opt['name']}]")
            
            # YOLO (预训练类型)
            if part == "1":
                if len(missing_yolos) == 0:
                    print("  -> 所有 YOLO 权重已就位，无需处理。")
                else:
                    print(f"  [提示] 仍有 YOLO 权重缺失。请手动前往 models/fabric-defect-detection/ 运行获取脚本。")
                    
            # Anomaly Detection models (PatchCore, RD4AD, EfficientAD)
            elif part in ["2", "3", "4"]:
                mid = opt["mid"]
                spec = reg.models[mid]
                tinfo = {
                    "m.patchcore": {
                        "run_cmd": [sys.executable, "main.py", "--baseline", "patchcore"],
                        "config_file": "models/anomaly_detection_for_textile_industry/config.yaml"
                    },
                    "m.rd4ad": {
                        "run_cmd": [sys.executable, "main.py", "--baseline", "rd4ad"],
                        "config_file": "models/anomaly_detection_for_textile_industry/config.yaml"
                    },
                    "m.efficientad": {
                        "run_cmd": [sys.executable, "main.py", "--baseline", "efficientad"],
                        "config_file": "models/anomaly_detection_for_textile_industry/config.yaml"
                    }
                }[mid]
                
                repo_abs_dir = WORKSPACE_ROOT / "models/anomaly_detection_for_textile_industry"
                
                # 准备 bottle 数据集
                dataset_root = repo_abs_dir / "data/mvtec"
                bottle_dir = dataset_root / "bottle"
                if not bottle_dir.exists():
                    print(f"  [INFO] 正在准备对齐训练数据集 (MVTec AD bottle)...")
                    try:
                        download_script = (
                            "from anomalib.data import MVTec; "
                            "dm = MVTec(root='./data/mvtec', category='bottle'); "
                            "dm.prepare_data()"
                        )
                        subprocess.run([sys.executable, "-c", download_script], cwd=str(repo_abs_dir), check=True)
                    except Exception as ex:
                        print(f"  [X] 下载数据集失败: {ex}。请确保已安装 anomalib。")
                        continue
                
                # 临时修改 config.yaml 并训练
                cfg_path = WORKSPACE_ROOT / tinfo["config_file"]
                original_cfg_content = cfg_path.read_text(encoding="utf-8")
                try:
                    import yaml
                    cfg_data = yaml.safe_load(original_cfg_content)
                    if "datamodule_configuration" in cfg_data:
                        cfg_data["datamodule_configuration"]["category"] = "bottle"
                        cfg_data["datamodule_configuration"]["root"] = "./data/mvtec"
                    cfg_path.write_text(yaml.safe_dump(cfg_data), encoding="utf-8")
                    
                    print(f"  [RUN] 执行命令: {' '.join(tinfo['run_cmd'])}")
                    subprocess.run(tinfo["run_cmd"], cwd=str(repo_abs_dir), check=True)
                    print(f"  [SUCCESS] [{opt['name']}] 训练完成！")
                    opt["status"] = "✔ 已就位"
                except subprocess.CalledProcessError as err:
                    print(f"  [X] 训练失败，错误码 {err.returncode}")
                finally:
                    cfg_path.write_text(original_cfg_content, encoding="utf-8")

            # MoECLIP
            elif part == "5":
                if not clip_base_ok:
                    print(f"  [X] 拒绝执行: 缺失只读基础骨干权重 ViT-L-14-336px.pt。请先手动准备该权重！")
                    continue
                
                repo_abs_dir = WORKSPACE_ROOT / "models/MoECLIP"
                run_cmd = [sys.executable, "train.py", "--dataset", "MVTec", "--epoch", "5", "--save_path", "ckpt/baseline"]
                
                print(f"  [RUN] 执行 MoECLIP 训练命令: {' '.join(run_cmd)}")
                try:
                    subprocess.run(run_cmd, cwd=str(repo_abs_dir), check=True)
                    print(f"  [SUCCESS] MoE-CLIP 门控头训练完成！")
                    opt["status"] = "✔ 已就位"
                except subprocess.CalledProcessError as err:
                    print(f"  [X] MoE-CLIP 训练失败，错误码 {err.returncode}")
        
        print("\n" + "="*50 + "\n")

    print("\n=== 前置检查与准备流程结束 ===")
    return 0
