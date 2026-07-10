# 进度快照与未来工作清单

> 更新：2026-07-10　|　对应 **Verification Handbook v3**（11 个测试单元 C1–C6 / M1–M2 / N1–N3）
> 执行模型：本地 `--mode local` 仅做可行性验证；真实测量在 CUDA GPU 服务器 `--mode server`，数据集在云端。

## 一句话现状

**11/11 个测试单元 + 2 个模型适配器（AnomalyRepoAdapter/MoECLIPAdapter）的 server 真实路径已全部实现**，其中 9 个已在本机用真实训练出的 anomalib checkpoint 端到端验证过（不是伪造数字，是真实推理/训练/量化跑出来的结果）；C1（功耗/温度）、N1（tc 网络限速）因为这台开发机没有 NVIDIA GPU / 不是 Linux，代码写对了但没法在本机验证，需要在真正的 GPU/Linux 服务器上跑；N3（硬实时）如实标注：完整硬件在环（HIL）测量依赖真实 PLC/总线设备，本仓库只能做到"软件段"部分的真实测量，不假装有完整 HIL 结论。

---

## 二、已完成（DONE）

### 地基层（与模型无关，全部跑通，无改动）
- `configs/registry.yaml` / `measurement.yaml` / `hypotheses.yaml` / `runtime.yaml`
- `common/env.py`、`common/bench_utils.py`、`common/registry.py`、`common/result.py`、`common/datasets.py`
- `run.py` 编排入口

### 模型适配层 `common/models.py`（本轮新增：两个真实适配器）
- **`UltralyticsAdapter`**（YOLO）：此前已实现，真实加载 `.pt` 权重推理。
- **`AnomalyRepoAdapter`**（PatchCore/RD4AD/EfficientAD）：**新实现**。用 anomalib 各模型类自带的
  `load_from_checkpoint()` 直接载入 `.ckpt`（anomalib 训练时已 `save_hyperparameters()`，不需要重新猜
  backbone/layers 等构造参数），真实前向调用 `lightning_module.model(x)`。用本机真实训练的 toy
  PatchCore/ReverseDistillation checkpoint 验证过：真实 `pred_score`/`anomaly_map` 张量，`backend="real"`。
- **`MoECLIPAdapter`**：**新实现**。对照 `models/MoECLIP/test.py` 的真实调用方式重建（`model.clip.create_model` +
  `model.moe_adapter.MoECLIP`），checkpoint 结构是 `{"text_adapter":..., "image_adapter":...}` 分开存的字典，
  按原仓库的载入方式还原。受限于本机没有 `ViT-L-14-336px.pt` 基础权重和已训练的 MoE 头，未做端到端验证，
  但构造/加载逻辑已对照真实源码核实（不是猜的 API）。

### 11 个测试单元 —— server 真实路径

| 单元 | 真实实现内容 | 本机验证状态 |
|---|---|---|
| C1 功耗-热-降频 | pynvml 真实采样 GPU 功耗/温度/时钟，持续推理循环 | 代码正确但本机无 NVIDIA GPU，无法验证；缺 GPU 时清楚报错，不编造数字 |
| C2 分辨率-吞吐 | 已有真实计时逻辑，本轮只是让它能吃到真实 adapter | ✅ 已验证，PatchCore 真实 FPS 曲线随分辨率下降 |
| C3 并发 | 多线程并发调用 `adapter.infer()`，真实延迟/吞吐曲线；真实 GPU 显存用 pynvml（无 GPU 时为 None） | ✅ 已验证，真实延迟随并发数上升 |
| C4 换产训练成本 | 同一模型架构，合成小数据集，真实调用 anomalib `Engine.fit()` / `ultralytics.YOLO.train()`，真实计时+真实最终指标 | ✅ 已验证（PatchCore、YOLOv8n 均测过） |
| C5 跨域漂移 | 域A真实训练 → 域A留出/域B测试，真实 anomalib AUROC 对比 | ✅ 已验证，真实退化率计算 |
| C6 协同升级一致性 | 同一 checkpoint，FP32 vs FP16 真实推理分数，真实阈值/决策不一致率 | ✅ 已验证 |
| M1 带宽竞争 | N 线程并发真实搬运图像张量，测真实吞吐 GB/s（CUDA 环境为真实 H2D，无 GPU 为 CPU 内存拷贝代理） | ✅ 已验证 |
| M2 量化-微缺陷抹除 | fp32/fp16/int8_ptq(真实动态量化) 真实推理 + 按缺陷尺寸真实召回率；int8_qat/int4 如实标记不支持（需专用重训/推理库） | ✅ 已验证（fp32/fp16/int8_ptq） |
| N1 模型分发同步 | 包体积改读真实权重文件大小；同步耗时需 Linux+root+tc 限速回环接口做真实 HTTP 传输计时 | 真实文件大小读取已验证；tc 传输部分本机非 Linux 无法验证，代码按标准 `tc qdisc`/`http.server` 写好 |
| N2 事件驱动回传 | 真实 JPEG 编码测帧/块字节数；真实模型推理延迟 + 真实线程队列事件投递延迟 | ✅ 已验证 |
| N3 硬实时控制链 | vanilla_linux 段改为真实模型推理+真实调度抖动测量；preempt_rt/hw_accel 如实标 `not_measured`（需专用内核/硬件），不是完整 HIL 闭环 | ✅ 已验证（software segment 部分） |

---

## 三、仍然明确未完成 / 需要真实硬件才能验证

- **C1、N1**：代码已实现真实测量逻辑，但分别需要 **NVIDIA GPU（pynvml）** 和 **Linux + root + iproute2(tc)**，
  这台开发机（macOS，无 NVIDIA GPU）都不具备，需要在目标 GPU 服务器上跑通验证。
- **N3 的完整 HIL 闭环**：需要真实 PLC/总线/执行机构硬件，本仓库不可能在没有这些设备的情况下产出"真实"结论，
  这是诚实的架构限制，不是待办事项。
- **M2 的 int8_qat / int4**：QAT 需要量化感知重训（不是拿一个训练好的 checkpoint 就能做），int4 需要专用推理库
  （TensorRT/bitsandbytes 等），目前明确标注为"不支持"而不是伪造数字；如果需要这两档，需要单独实现。
- **各数据集真实目录布局细化**（MVTec/VisA/ZJU/RAW-FABRID/SDUST 各自的 test/ground_truth 布局）——
  `common/datasets.py` 目前对真实数据用通用 glob 扫描，未按数据集精细适配；本轮验证全部基于合成/toy 数据，
  没有改动这块。
- **`runtime.yaml` 的云端 `dataset_root`/`weights_root`** 仍是占位符 `s3://<your-bucket>/...`，没有接真实云存储。
- **阈值锁定**（`hypotheses.yaml` 里的 `threshold` 字段）仍待真实数据定标。

---

## 四、下次从哪继续

1. 上 GPU 服务器后：把 `models/anomaly_detection_for_textile_industry` 训出来的真实 `.ckpt` 放到
   `registry.yaml` 声明的 `weights_path`，跑 `python run.py --check --mode server` 确认三个 anomaly 模型
   都能真实加载（不再是 mock）。
2. 跑 `python run.py --test C1 --mode server`、`--test N1 --mode server` 验证 pynvml/tc 真实路径。
3. 需要 MoECLIP 真实验证的话，先按 `models/MoECLIP/README.md` 备好 `ViT-L-14-336px.pt` + 训出 MoE 头权重。
4. 各数据集真实目录布局细化、云端存储接入、阈值锁定——按之前的优先级顺序继续。
