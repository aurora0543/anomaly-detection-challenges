#!/usr/bin/env python3
"""子项目一键初始化与补丁应用脚本。

当从 GitHub 克隆本项目后，运行此脚本可以自动：
  1. 初始化并拉取所有 Git 子模块（软链接仓库的源代码）
  2. 将 `custom_patches/` 目录下的自定义修复与配置文件（如 RD4AD 的修复）覆盖应用到对应的子项目里。
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

def run_command(cmd: list, cwd: Path = None):
    print(f"[RUN] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if result.returncode != 0:
        print(f"[ERROR] 命令执行失败 (Exit Code {result.returncode}): {' '.join(cmd)}")
        sys.exit(result.returncode)

def main():
    root_dir = Path(__file__).resolve().parent
    
    # 1. 初始化并拉取子模块
    print("=== 第一步：初始化并更新 Git 子模块 ===")
    run_command(["git", "submodule", "update", "--init", "--recursive"], cwd=root_dir)
    print("[SUCCESS] 子模块同步完成！\n")

    # 2. 应用自定义补丁文件
    print("=== 第二步：应用 custom_patches 中的自定义补丁 ===")
    patch_src_root = root_dir / "custom_patches"
    subprojects_dest_root = root_dir / "models"

    if not patch_src_root.exists():
        print("[INFO] 未找到 custom_patches/ 目录，跳过补丁应用。")
        return

    # 递归复制补丁文件到目标位置
    for src_file in patch_src_root.rglob("*"):
        if src_file.is_file():
            # 计算相对于 custom_patches 的路径
            rel_path = src_file.relative_to(patch_src_root)
            dest_file = subprojects_dest_root / rel_path
            
            # 创建目标父目录
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            
            print(f"应用补丁: {rel_path} -> models/{rel_path}")
            shutil.copy2(src_file, dest_file)
            
    print("[SUCCESS] 所有自定义修改与补丁已成功应用！")

if __name__ == "__main__":
    main()
