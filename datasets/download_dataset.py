#!/usr/bin/env python3
"""数据集一键下载与解压脚本 —— 供云端/服务器自动化准备数据。

支持模式：
  1. MVTec AD: 自动拉取官方 Tarball 并解压
  2. VisA: 自动拉取官方 S3 备份 Tarball 并解压
  3. Hugging Face Datasets: 通过指定 --hf-repo (例如 aurora0543/raw-fabrid-mvtec)，一键用 API 下载自定义数据集。

依赖需求:
  pip install requests tqdm huggingface_hub fiftyone
"""

import os
import sys
import argparse
import tarfile
import urllib.request
from pathlib import Path

# 官方公开直链
VISA_URL = "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_pytorch.tar"

def download_with_progress(url: str, dest_path: Path):
    """带进度条的下载函数。"""
    try:
        from tqdm import tqdm
        import requests
    except ImportError:
        print("[INFO] 未安装 requests 或 tqdm，将使用标准 urllib 下载（无进度显示）...")
        print(f"正在从 {url} 下载...")
        urllib.request.urlretrieve(url, str(dest_path))
        print("下载完成！")
        return

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024  # 1MB
    
    with open(dest_path, "wb") as f, tqdm(
        desc=dest_path.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            size = f.write(data)
            bar.update(size)

def extract_tar(archive_path: Path, dest_dir: Path):
    """解压 tar/tar.xz 压缩包。"""
    print(f"正在解压 {archive_path.name} 到 {dest_dir} ...")
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:*") as tar:
        tar.extractall(path=str(dest_dir))
    print("解压完成！")

def handle_mvtec(dest_root: Path):
    try:
        import fiftyone as fo
        import fiftyone.utils.huggingface as fouh
    except ImportError:
        print("[ERROR] 使用 MVTec AD 模式需要先安装 fiftyone 包：")
        print("        pip install fiftyone")
        sys.exit(1)

    print("=== 开始通过 FiftyOne 加载 MVTec AD 数据集 ===")
    dataset = fouh.load_from_hub("Voxel51/mvtec-ad")
    fo.launch_app(dataset)

def handle_visa(dest_root: Path):
    target_dir = dest_root / "VisA"
    archive_file = dest_root / "VisA_pytorch.tar"
    
    if (target_dir / "bottle").exists() or (target_dir / "cashew").exists():
        print(f"[INFO] VisA 数据集已存在于 {target_dir}，跳过下载。")
        return

    print("=== 开始下载 VisA 数据集 (约 1.5 GB) ===")
    download_with_progress(VISA_URL, archive_file)
    extract_tar(archive_file, target_dir)
    
    if archive_file.exists():
        os.remove(archive_file)
        print("清理临时压缩包完成。")

def handle_huggingface(repo_id: str, dest_dir: Path):
    """通过 Hugging Face API 自动同步自定义数据集（如 RAW_FABRID, ZJU-Leaper）。"""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[ERROR] 使用 Hugging Face 模式需要先安装 huggingface_hub 包：")
        print("        pip install huggingface_hub")
        sys.exit(1)

    print(f"=== 开始从 Hugging Face 同步数据集: {repo_id} ===")
    print(f"目标下载目录: {dest_dir}")
    
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(dest_dir),
            local_dir_use_symlinks=False
        )
        print("[SUCCESS] Hugging Face 数据集同步成功！")
    except Exception as e:
        print(f"[ERROR] 同步失败: {e}")
        print("提示: 如果该仓库是私有仓库，请先在终端运行 `huggingface-cli login` 进行授权。")

def main():
    parser = argparse.ArgumentParser(description="工业视觉异常检测挑战 - 数据集一建获取工具")
    parser.add_argument(
        "--dataset",
        choices=["mvtec", "visa", "hf"],
        required=True,
        help="选择获取哪个数据集：mvtec(官方下载) | visa(官方下载) | hf(通过 Hugging Face 同步自定义数据集)"
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=None,
        help="当 --dataset 为 hf 时，指定 Hugging Face 仓库名（如 aurora0543/raw-fabrid）"
    )
    parser.add_argument(
        "--dest",
        type=str,
        default=None,
        help="指定解压/下载的目标文件夹名称（默认存在当前脚本所在文件夹的对应同名文件夹中）"
    )
    args = parser.parse_args()

    # 确定下载目标文件夹（默认为本脚本所在 datasets 目录）
    script_dir = Path(__file__).resolve().parent
    
    if args.dataset == "mvtec":
        handle_mvtec(script_dir)
    elif args.dataset == "visa":
        handle_visa(script_dir)
    elif args.dataset == "hf":
        if not args.hf_repo:
            print("[ERROR] Hugging Face 模式必须提供 --hf-repo 仓库名！例如:")
            print("        python download_dataset.py --dataset hf --hf-repo aurora0543/raw-fabrid-small --dest RAW_FABRID_small")
            sys.exit(1)
        
        # 确定存放的子文件夹名
        sub_dir_name = args.dest if args.dest else args.hf_repo.split("/")[-1]
        target_path = script_dir / sub_dir_name
        handle_huggingface(args.hf_repo, target_path)

if __name__ == "__main__":
    main()
