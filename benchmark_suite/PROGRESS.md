# 进度快照与未来工作清单

> 更新：2026-07-07　|　对应 **Verification Handbook v3**（11 个测试单元 C1–C6 / M1–M2 / N1–N3）
> 执行模型：本地 `--mode local` 仅做可行性验证；真实测量在 CUDA GPU 服务器 `--mode server`，数据集在云端。

## 一句话现状

**11/11 个测试单元全部实现**（计算 C1–C6 + 内存 M1/M2 + 通信 N1/N2/N3），地基（脚手架 + 执行模型 + 模型适配层骨架 + 数据接口）已完成。所有单元 local 可行性跑通、核心逻辑单测通过；剩余工作是把各单元的 **server 真实路径**接上（真实模型/量化/训练/网络/HIL）+ 报告层 + 阈值锁定。

---

## 二、已完成（DONE）

### 地基层（与模型无关，全部跑通）
- `configs/registry.yaml` — 硬件/7模型/5数据集/11测试单元 + 兼容矩阵（§2.2–2.4）
- `configs/measurement.yaml` — warmup/iters/repeats/percentiles/bootstrap（§2.5）
- `configs/hypotheses.yaml` — 假设 H1–H11 + 判据类型 + 阈值占位（§1.3）
- `configs/runtime.yaml` + `common/runtime.py` — 执行档 local/server（§2.8）
- `common/env.py` — 环境元数据采集（torch/pynvml 可选）
- `common/bench_utils.py` — Timer / run_timed / summarize / bootstrap_ci
- `common/registry.py` — 注册表加载 + 一致性校验
- `common/result.py` — 统一 result.json 契约（构建/校验/落盘）
- `common/models.py` — 统一适配接口 + MockModel 回退 + 工厂 + feasibility_check
  - `UltralyticsAdapter`（YOLO）真实载入路径已写；anomaly / MoECLIP 接口就位、真实对接标 TODO
- `common/datasets.py` — 统一 Sample 流 + 合成回退 + 真实扫描；`defect_size_bins`（M2用）、`split_by_domain`（C5用）
- `run.py` — 编排入口：`--env/--validate/--list/--status/--check/--test/--demo-result/--mode`

### 已实现的测试单元（7/11）——计算维度全部完成 + 内存 M2

所有单元均：核心判定逻辑可单测（已通过）+ local 合成演示走通 + server 真实路径标 TODO + result.json 契约。

- **C1 功耗-热-降频**（`compute/c1_power_thermal.py`，H1）：`analyze_fps_series`/`judge_h1`。local 演示持续 22 FPS<节拍30 → supported。待补：power.py 真实采样。
- **C2 分辨率-吞吐**（`compute/c2_resolution_sweep.py`，H2）：`loglog_slope`/`judge_h2`。注意力−3.80 vs 卷积−2.00 → supported。待补：thop GFLOPs、profiler 分层、带宽。
- **C3 并发**（`compute/c3_concurrency.py`，H4）：`max_streams_under_budget`/`judge_h4`。异常组 2 路 vs 监督组 32 路 → supported。待补：真实多进程/多流。
- **C4 换产训练**（`compute/c4_changeover_training.py`，H3）：`full_vs_incremental_speedup`/`judge_h3`。全量 2h、边/云 20x → supported。待补：真实训练到收敛。
- **C5 跨域漂移**（`compute/c5_domain_shift.py`，H5）：`degradation_pct`/`judge_h5` + 真实 `split_by_domain`。退化 19% → supported。待补：真实域A训→域B测。
- **C6 协同升级一致性**（`compute/c6_upgrade_consistency.py`，H6）：`optimal_threshold`/`decision_disagreement`/`judge_h6`。沿用旧阈值不一致 9.5%、重标定后 0% → supported。待补：真实双版本推理。
- **M2 量化-微缺陷抹除**（`memory/m2_quantization.py`，H8）：`recall_by_size`/`micro_recall_drop`/`judge_h8`。micro 召回随量化降最多 → supported。待补：真实量化导出与推理。

---

## 三、未完成（TODO）——完整工作清单

> 说明：**11 个测试单元的骨架与 local 可行性逻辑已全部完成**（B/C/D 段的"写单元"已做完）。
> 下面清单里，测试单元部分现在的含义是"接 server 真实路径"；A/E 段（通用能力与收尾）仍是主要待办。
> 优先级：P1=先做（打通端到端产出），P2=核心测试单元真实路径，P3=依赖硬件或较重。

### A. 通用能力（多个测试单元共用）
- [x] **A1【已完成】报告层 `common/report.py`** — 聚合全部单元（内存跑 or 读 results/）→ 逐假设判定总表（按维度分组+计数）→ `results/report.md`。入口 `run.py --report [--source memory|disk]`。（曲线图待后续可选加。）
- [ ] **A2【P2】功耗/资源采样 `common/power.py`** — pynvml（GPU 功耗/温度/显存/带宽）、tegrastats（Jetson）、`nvidia-smi -pl` 功耗封顶封装。C1/M1 前置。
- [ ] **A3【P2】真实模型对接 `AnomalyRepoAdapter`** — 将 anomaly 仓库上 PYTHONPATH，按 config.yaml 载入 PatchCore/RD4AD/EfficientAD（复用其 `src/anomaly_pipeline.py`、`inference.py`、ONNX 导出）。
- [ ] **A4【P2】真实模型对接 `MoECLIPAdapter`** — import 其 `model/` 包，载入 OpenCLIP ViT-L-14-336px 权重，接 forward。
- [ ] **A5【P2】量化导出 `adapter.export/infer_quantized`** — TensorRT INT8 calibrator / ONNXRuntime static quant / `torch.ao` QAT。M2 server 真实路径依赖。
- [ ] **A6【P3】各数据集真实目录布局细化** — MVTec/VisA/ZJU/RAW-FABRID/SDUST 各自的 test/ground_truth 布局适配（现为通用 glob）。
- [ ] **A7【P1】云端数据/权重挂载** — 填 `runtime.yaml` 的 `dataset_root/weights_root`，实现 server 从云端拉数据。

### B. 计算维度测试单元（还差 5 个）
- [ ] **C1【P2】端侧资源受限：功耗-热-降频**（H1，依赖 A2）— 30min 持续负载，FPS 衰减/温度/功耗/降频时间；无边缘设备走 GPU 功耗封顶 proxy。
- [ ] **C2 收尾【P2】** — 接 thop GFLOPs、torch.profiler 分层耗时、显存带宽（现斜率主逻辑已完成）。
- [ ] **C3【P2】扩展性：多实例并发**（H4）— N=1..32 并发，每路延迟、总吞吐、GPU/CPU 利用率，找延迟翻倍点/吞吐饱和点。
- [ ] **C4【P2】训练门槛：换产训练成本**（H3，较重）— 全量训到收敛耗时、增量微调、峰值显存、能耗；ZJU 按纹理型划旧/新。
- [ ] **C5【P2】数据异构：跨域漂移**（H5，依赖 `split_by_domain`）— 单域 vs 跨域精度退化；联邦仅单轮 FedAvg 探针。
- [ ] **C6【P2】生命周期运维：协同升级一致性**（H6）— 引擎/权重/后处理分别升级，测阈值漂移、灰度/回滚耗时、升级前后结论一致性。

### C. 内存维度测试单元（还差 1 个 + M2 收尾）
- [ ] **M1【P2】HBM 带宽竞争**（H7，依赖 A2）— resident vs 端到端（含 H2D/D2H）吞吐差、数据搬运占比、带宽利用率；256 vs 2048、batch 1 vs 8。
- [ ] **M2 收尾【P2】**（依赖 A5）— server 真实量化评估替换合成评估器。

### D. 通信维度测试单元（还差 3 个）
- [ ] **N1【P2】模型分发与同步**（H9）— 全量/差分/结构化/蒸馏更新 × `tc` 模拟 LAN/工厂/恶劣网络；同步耗时 P50/P95/P99、带宽节省；异常检测（含 memory bank）包体积对比。
- [ ] **N2【P2】事件驱动回传**（H10）— 全量 vs 事件 vs 分级回传；注入 1% 缺陷帧，埋点 T1–T5 测检出→云端延迟、省带宽比、误报额外流量。
- [ ] **N3【P3】硬实时控制链**（H11，需 HIL/PLC 硬件）— 端到端控制延迟/抖动，vanilla vs PREEMPT_RT vs 硬件加速；无硬件时标 `待验证`，只做软件段延迟画像。

### E. 收尾与交付
- [ ] **E1【P1】阈值锁定** — 在 `hypotheses.yaml` 填真实门槛（产线节拍 FPS、硬实时预算、高危回传时限等）。
- [ ] **E2【P2】server 端到端跑一遍** — 装 torch/ultralytics/pynvml/tensorrt，真实数据，产出全部 result.json。
- [ ] **E3【P2】汇总报告生成** — 用 A1 的 report.py 出最终「逐假设判定」总表 + 曲线图。
- [ ] **E4【P3】单元测试与复现** — 为各核心逻辑补 pytest；固定随机种子与环境记录。

---

## 四、进度表（对照 11 条挑战）

| 维度 | 测试 | 假设 | 状态 | 备注 |
|---|---|---|---|---|
| 计算 | C1 功耗-热-降频 | H1 | ◑ 主逻辑完成 | 待接 power.py 真实采样；无边缘走 proxy |
| 计算 | C2 分辨率-吞吐 | H2 | ◑ 主逻辑完成 | 待接 GFLOPs/profiler/带宽 |
| 计算 | C3 并发 | H4 | ◑ 主逻辑完成 | 待接真实多进程/多流 |
| 计算 | C4 换产训练 | H3 | ◑ 主逻辑完成 | 待接真实训练到收敛 |
| 计算 | C5 跨域漂移 | H5 | ◑ 主逻辑完成 | split_by_domain 已用；待真实域A训→域B测 |
| 计算 | C6 协同升级 | H6 | ◑ 主逻辑完成 | 待接真实双版本推理 |
| 内存 | M1 带宽竞争 | H7 | ◑ 主逻辑完成 | 多工位采集×总线利用率；待真实带宽采样 |
| 内存 | M2 量化抹除 | H8 | ◑ 主逻辑完成 | 待接真实量化 |
| 通信 | N1 模型同步 | H9 | ◑ 主逻辑完成 | 包体积×网络；待真实 tc 网络 |
| 通信 | N2 事件回传 | H10 | ◑ 主逻辑完成 | 待接真实视频流埋点 |
| 通信 | N3 硬实时 | H11 | ◑ 主逻辑完成 | 无 HIL 标 pending_hardware，仅软件段画像 |

图例：☐ 未写｜◑ 主逻辑完成（local 可跑、server 真实路径待接）｜✔ 完全完成

**全部 11 个测试单元均已 ◑ 主逻辑完成**（local 可行性跑通、核心逻辑单测通过）。剩余为各单元 server 真实路径对接 + 通用能力（报告层/功耗采样/真实模型/量化/云端数据）+ 收尾。

---

## 五、下次从哪继续

建议顺序：**A1 报告层**（先打通"多结果→汇总判定"，让已完成的 C2/M2 立即产出可读结论）→ 再按维度补测试单元（C3 → M1 → N1/N2 …）→ A2/A3/A4/A5 真实对接（上服务器前）→ E 系列收尾。

每个测试单元都遵循 §3 统一模板与 `common/*` 现有工具，新增时只需：写 `<dim>/<id>_xxx.py`、在 `run.py` 的 `TEST_DISPATCH`/`IMPLEMENTED` 注册、`--test <ID>` 自测。
