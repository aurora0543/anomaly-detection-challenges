# 项目工作同步 / Project Status

> 这个文件是给**下一个接手的人或 AI Agent**看的,不管你是谁、什么时候打开这份文件,读完这一份应该能知道
> "现在整体做到哪一步了、哪些是真的、哪些还是占位、下一步该干什么"。请在做完一段有意义的工作后更新这个文件,
> 而不是等到最后才补——这个项目预计会有多个 Agent/协作者交替接手。
>
> 最后更新:2026-07-10

---

## 一、这个项目是什么

工业布匹表面缺陷检测的模型验证平台。核心问题:同一批数据(ZJU-Leaper 公开数据集,`pattern1` 纹理类别)下,
对比几种不同范式的检测方案——传统监督目标检测(YOLO 系列)vs 无监督异常检测(PatchCore/RD4AD/EfficientAD/
SuperSimpleNet)vs 零样本异常检测(MoECLIP)——在精度、速度、训练成本、部署鲁棒性等维度谁更合适,并把这套
对比方法沉淀成一个可复用的"验证平台"(`benchmark_suite/`),而不是一次性脚本。

## 二、仓库结构与状态总览

```
anomaly-detection-challenges/          (根仓库, GitHub: aurora0543/anomaly-detection-challenges)
├── models/
│   ├── anomaly_detection_for_textile_industry/   [submodule, 私有 fork]  ← PatchCore/RD4AD/EfficientAD/SuperSimpleNet
│   ├── fabric-defect-detection/                  [submodule, 私有 fork]  ← YOLOv8n/v8s/v11n
│   └── MoECLIP/                                  [submodule, 私有 fork]  ← 零样本异常检测，未训练
├── datasets/ZJU-Leaper                           原始数据集（外置存储，符号链接）
├── benchmark_suite/                              验证平台框架（见 benchmark_suite/PROGRESS.md 详细版）
└── analysis/                                     轻量级跨模型对比+并发测试结果（本次新增，独立于 benchmark_suite）
```

三个 submodule 都已经从原作者的公开仓库改成指向 `aurora0543` 账号下的私有仓库（详见根仓库 PR #1），
**当前是 public 仓库**（用户为了免认证部署故意设的，见对话记录，不是遗漏）。

### 模型训练状态（截至本次更新）

| 模型 | 状态 | 权重位置 | 精度来源 |
|---|---|---|---|
| PatchCore | ✅ 已训练 | `anomaly_detection_for_textile_industry/checkpoints/last.ckpt`（**注意：没有按模型命名**，见下方"已知问题"） | `results/results/EfficientAd/report/20260709_210933_*.txt` |
| RD4AD (ReverseDistillation) | ✅ 已训练 | `checkpoints/ReverseDistillation-latest.ckpt` | `.../20260709_214741_*.txt` |
| EfficientAD | ✅ 已训练 | `checkpoints/EfficientAd-latest.ckpt` | `.../20260710_110726_*.txt`（第一次因为服务器连不上 HuggingFace 失败过，换 `HF_ENDPOINT` 镜像后成功） |
| SuperSimpleNet | ✅ 已训练 | `checkpoints/last-v2.ckpt`（同样没按模型命名） | `.../20260709_231222_*.txt` |
| YOLOv8n | ✅ 已训练 | `fabric-defect-detection/weights/YOLOv8n.pt` | `results/yolov8n_metrics.json` |
| YOLOv8s | ✅ 已训练 | `weights/YOLOv8s.pt` | `results/yolov8s_metrics.json` |
| YOLOv11n | ✅ 已训练 | `weights/YOLOv11.pt` | `results/yolov11n_metrics.json` |
| MoECLIP | ⏸ 用户决定暂不训练 | — | — |

全部（除 MoECLIP）都在 ZJU-Leaper `pattern1` 这同一份数据集/同一批测试图上跑的，精度可以直接横向比较。

## 三、`benchmark_suite/`（正式验证框架）现状

详见 [`benchmark_suite/PROGRESS.md`](benchmark_suite/PROGRESS.md)，这里只摘要：

- 对应 `Verification Handbook v3` 的 11 个测试单元（C1-C6/M1-M2/N1-N3）+ 2 个模型适配器，
  **9/11 单元已经是真实实现并验证过**（不是 mock，是真的用训练出来的 checkpoint 跑推理/训练/量化）。
- C1（功耗/温度）、N1（tc 网络限速）代码写对了，但分别需要 **NVIDIA GPU + pynvml** 和 **Linux+root+tc**，
  这两样这台开发机都没有，得在真正的 GPU/Linux 服务器上验证。
- N3（硬实时）如实标注：真正的硬件在环（HIL）需要 PLC/总线设备，我们没有，只做了"软件段"的真实测量。
- **本次新增的关键修复**：`common/runtime.py` 的 `resolve_device()` 之前只认 CUDA，在 Apple Silicon（MPS 也是
  真实 GPU）上会被误判成"没有 GPU"而退化到 CPU——这是一个真实 bug，已修复，`runtime.yaml` 的 `server` 档
  现在会按 `cuda → mps → 报错` 的顺序选设备。`common/env.py` 的 `collect_env()` 同理修了，之前也测不到 MPS。
- **本次新增**：`registry.yaml` 补了 `m.supersimplenet`（之前完全没注册，训了但验证平台不认识它），并且给
  PatchCore/SuperSimpleNet 的 checkpoint 建了符号链接对齐 `weights_path` 期望的文件名
  （`Patchcore-latest.ckpt` → `last.ckpt`，`Supersimplenet-latest.ckpt` → `last-v2.ckpt`，**这两个符号链接是本机
  本地建的，没有提交到 git（checkpoint 本来就被 gitignore），换一台机器/换服务器要重新建**）。

## 四、`analysis/`（本次新增，独立脚本）——本机全部验证已跑完

跟 `benchmark_suite` **是两条并行、目前没有合并的线**——这个脚本更直接：精度直接从上面表格里那些评估报告/
metrics.json 解析，不重新跑；速度/参数量/并发吞吐在本机（Apple Silicon, MPS）现场真实测量，保证同一硬件下的
横向对比公平。**结果已经跑完并汇总在 [`analysis/results/summary.md`](analysis/results/summary.md)**，包含：

- 单张推理延迟/参数量/精度对比（7 个模型：PatchCore/RD4AD/EfficientAD/SuperSimpleNet/YOLOv8n/v8s/v11n）
- 并发吞吐测试（复用 `benchmark_suite` 的真实 C3 实现，见下方"重要修复"）

跑法：`python analysis/compare_models.py`（单张推理对比）；并发测试用
`python benchmark_suite/run.py --test C3 --mode server`（**必须在仓库根目录跑**）。

**本次新增**：C2/C4/C5/C6/M1/M2/N2 共 7 个测试单元也已在全部 7 个真实模型上补跑完（`analysis/run_remaining_tests.py`
+ `analysis/rerun_m2_n2_only.py`，结果在 `analysis/results/remaining_tests.json`），32 个组合 27 个成功、
2 个跳过（EfficientAD 的 C4/C5，需联网下载约 1.5GB Imagenette，用户选择不等）、2 个真实失败（EfficientAD 的
C6/M2，FP16 转换架构本身限制）。详见 [`benchmark_suite/PROGRESS.md`](benchmark_suite/PROGRESS.md) 第六节和
[`analysis/results/summary.md`](analysis/results/summary.md) 第四节。

已知局限：**没有云端 CUDA 的速度数据可比较**——检查过云端训练日志（ultralytics 的 `results.csv` 没有速度列，
anomalib 的 `Engine` 调用用的是 `logger=False`），云端训练时从没把单张推理耗时记录下来，所以现在只能是
"MPS 本机内部横向对比"，不是"云端 vs 本机"对比。如果要做这个对比，得专门在云服务器上跑一次
`analysis/compare_models.py`（脚本本身不需要改，torch 会自动选 CUDA）。

### 重要修复：PyTorch MPS 后端的线程安全问题（真实踩坑，不是假设）

`benchmark_suite` 的 C3 并发测试原本用"每路并发开一个 Python 线程"模拟，这个方式在 CPU 上没问题，但**在这台
Mac 的 MPS 后端上会直接原生崩溃**（`-[_MTLCommandBuffer commit]` 段错误，退出码 139，用一个最小复现脚本稳定
复现过，不是偶发）——多个线程同时往 Metal 提交 command buffer 不是线程安全的操作。

已修复（`benchmark_suite/compute/c3_concurrency.py`）：CPU 上继续用真线程（多核确实能并行，这个测的是真的），
MPS/CUDA 上改成单线程、把并发数当 batch size 处理（这也更贴近真实推理服务用动态批处理支撑并发请求的做法，
不是退而求其次）。同时修了 `common/runtime.py`（`resolve_device()` 之前只认 CUDA，MPS 会被误判成"没有
GPU"退化到 CPU）、`common/env.py`（`collect_env()` 同理）、`common/models.py`
（`UltralyticsAdapter` 之前从不显式挪到 GPU/MPS，YOLO 系列一直在 CPU 上跑）。

## 五、已知问题 / 下次要注意的坑

1. **checkpoint 命名不规范**：PatchCore/SuperSimpleNet 训练时没有走到 `main.py` 里
   `rename_run_and_update_symlink` 那一步给出规范文件名（可能是训练中途被打断，或者 checkpoint 目录里混了
   Lightning 自动存的 `last.ckpt`/`last-vN.ckpt`）。**下次训练这两个模型时，注意 `config.yaml` 里
   `paths.checkpoint_destination` 是否按模型分开设置**，否则每次都要靠文件大小 + 试加载来倒推是哪个模型。
2. **`run.py` 必须从仓库根目录跑**，不能在 `benchmark_suite/` 目录里跑——`registry.yaml` 里的 `weights_path`
   是相对仓库根目录写的相对路径，cwd 不对会报"权重缺失"但其实文件是在的。
3. MoECLIP 还没训练（用户明确说"先不做"），`registry.yaml` 里的 `m.moeclip` 条目和 `MoECLIPAdapter` 都已经
   实现好了，权重一旦训出来直接能用。

## 六、下一步可选项（未做，等指示）

- 在真正的 GPU/Linux 云服务器上补跑 C1（功耗/温度）、N1（tc 网络限速）两个还没法在本机验证的测试单元。
- 训练 MoECLIP（需要先手动放好 `ViT-L-14-336px.pt` 基础权重）。
- 决定 `analysis/compare_models.py` 和 `benchmark_suite` 要不要合并，还是保持两条独立的线（一条正式留痕、
  一条轻量快查）——目前两边都能用，互不冲突，先搁置。
