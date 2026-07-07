# 工业视觉异常/缺陷检测 —— 开销挑战实测验证手册 v3

> 面向自动化实现的验证手册（布匹瑕疵检测为主实例）
> 对应文档：《工业异常检测及布匹瑕疵应用中的挑战 v2》第 2.2 节「表 4 布匹缺陷检测开销挑战汇总」
> 版本：v3　|　整理：Lee　|　日期：2026-07-07

---

## 0. 如何使用本手册（给读者，也给后续的代码生成器）

本手册有两个读者：**你**（需要读懂并据此推进导师任务）和**后续的代码生成流程**（需要据此生成一批可运行的自动化脚本）。为此，全书遵循以下约定：

1. **一切"实体"先注册、后引用。** 模型、数据集、硬件都在第 2 章以「注册表」形式集中定义，每个实体有唯一 `id`。后面的测试单元只引用 `id`，不重复描述。生成代码时，注册表直接对应配置文件（`config.yaml`）里的字典。
2. **每个测试是一个"测试单元"（Test Unit），结构固定。** 见 §3 的统一模板：`目的 → 对应挑战/假设 → 输入契约 → 被测组合 → 参数 → 测量指标 → 工具与 API → 执行步骤 → 输出契约(JSON) → 判据 → 风险与代理`。字段名固定，便于逐单元生成一个 Python 模块。
3. **结论以"假设"形式登记，允许证伪。** 不预设"挑战一定成立"。每条挑战对应一个可证伪假设 `H*`（§1.3），跑完填「支持 / 部分支持 / 不支持」。
4. **区分"实测"与"代理(proxy)"。** 缺少边缘硬件时，用 GPU 功耗封顶等手段做代理测量，结果必须带 `measurement_type: proxy` 标记，不与端侧实测混为一谈。
5. **判据分两类：硬约束门槛 vs 文献先验。** 只有产线硬约束（如实时帧率）作为"通过/不通过"门槛；其余预测值仅作对照参考，以实测值为准。

> 术语：本手册用「通用术语」书写以便迁移到其他工业场景；括号内以「布匹实例」给出纺织场景的具体化。例如：检测工位（验布机 / 机台）、产品换型（换布种 / 换产）。

---

## 1. 总览

### 1.1 验证目标

将表 4 中关于**计算 / 内存 / 通信**三类开销的**定性论述**，通过**可复现的量化实测**逐条检验其是否成立，形成证据链。产出为：每个测试单元一份结构化结果（JSON）+ 汇总报告。

### 1.2 挑战 → 测试单元 → 假设 索引表

| 维度 | 挑战（表4） | 测试单元 ID | 对应假设 |
|---|---|---|---|
| 计算 | 端侧资源受限（功耗墙/无风扇散热） | **C1** 持续负载功耗-热-降频 | H1 |
| 计算 | 注意力机制高复杂度（分辨率-吞吐冲突） | **C2** 分辨率-吞吐扫频 | H2 |
| 计算 | 扩展性瓶颈（多工位并发） | **C3** 多实例并发压力 | H4 |
| 计算 | 训练硬件门槛（换产重训/边缘增量） | **C4** 换产训练成本 | H3 |
| 计算 | 数据分布异构性（跨工位漂移） | **C5** 跨域泛化与漂移 | H5 |
| 计算 | 生命周期运维（协同升级） | **C6** 协同升级一致性 | H6 |
| 内存 | HBM 容量与带宽瓶颈 | **M1** 带宽竞争与数据搬运 | H7 |
| 内存 | 存储层次约束（低位宽量化抹除） | **M2** 量化-微缺陷抹除 | H8 |
| 通信 | 分布式版本同步 | **N1** 模型分发与同步 | H9 |
| 通信 | 非对称数据流控（高危异步回传） | **N2** 事件驱动回传延迟 | H10 |
| 通信 | 硬实时控制链协同 | **N3** 端到端控制延迟 | H11 |

> 共 11 个测试单元、11 条假设，与表 4 的 11 条挑战一一对应，无遗漏。

### 1.3 假设登记表（结果留空，允许证伪）

判据类型：`hard`＝产线硬约束门槛（决定通过/不通过）；`prior`＝文献先验（仅对照）；`mixed`＝两者兼有。

| ID | 假设（如为真则挑战成立） | 判据类型 | 判据 / 门槛（占位，运行前锁定） | 结果 |
|---|---|---|---|---|
| H1 | 无风扇/功耗封顶下持续实时性无法保障 | hard | 持续 FPS ≥ 产线节拍要求（实例：≥30 FPS） | ☐ 待填 |
| H2 | 含注意力模型在高分辨率下吞吐下降显著快于卷积模型 | prior | 注意力组吞吐-分辨率曲线斜率 > 卷积组，差异跨越 CI | ☐ 待填 |
| H3 | 换产重训延迟高、边缘难独立完成 | mixed | 全量训练→收敛耗时；边/云训练效率比 | ☐ 待填 |
| H4 | 异常检测模型单卡并发承载低于监督模型 | prior | 延迟≤节拍时单卡最大路数 | ☐ 待填 |
| H5 | 跨域（跨工位）精度显著退化 | prior | 跨域指标 − 单域指标 的下降幅度 | ☐ 待填 |
| H6 | 组件协同升级引发阈值漂移与检测不一致 | mixed | 升级前后同一标定集的结论差异率 | ☐ 待填 |
| H7 | 高分辨率下内存带宽成为瓶颈、数据搬运占比高 | prior | 带宽利用率峰值；H2D/D2H 占端到端比例 | ☐ 待填 |
| H8 | 低位宽量化抹除微弱缺陷特征 | mixed | 微缺陷(<10px)召回率下降幅度 | ☐ 待填 |
| H9 | 异常检测模型同步开销高于监督模型 | prior | 更新包体积、多机同步耗时 | ☐ 待填 |
| H10 | 事件触发回传显著省带宽，但需保证高危毫秒级响应 | hard | 省带宽比；高危事件回传延迟 | ☐ 待填 |
| H11 | 纯软件方案难以满足硬实时确定性 | hard | 端到端延迟、抖动（P99/最大值） | ☐ 待填 |

---

## 2. 全局约定（代码生成的核心依据）

> 本章的所有注册表建议 1:1 落成一个中央配置文件 `configs/registry.yaml`。测试脚本从这里读取实体定义，不硬编码。

### 2.1 硬件平台注册表

| id | 设备 | 角色 | 状态 | 处理约定 |
|---|---|---|---|---|
| `hw.cloud` | RTX 5090 / A100 | 云端训练与基准 | 已具备 | 全部基准测试、全量训练；`measurement_type=real` |
| `hw.edge` | Jetson Orin NX (16GB, 100 TOPS) | 边缘推理代表 | 未到位（建议采购） | 若到位则端侧实测；否则由 `hw.cloud` 功耗封顶代理 |
| `hw.tiny` | Raspberry Pi 4B + Coral TPU | 端侧极限/无风扇 | 未到位 | 同上；相关结论 `measurement_type=proxy` |

**代理约定（proxy convention）：** 无 `hw.edge` 时，在 `hw.cloud` 上：
- 功耗封顶：`nvidia-smi -pl <watts>`（如封顶到 30/50/70 W 模拟不同 TDP 档）
- 锁频：`nvidia-smi -lgc <minMHz,maxMHz>`，`-lmc` 锁显存频
- 结果字段 `measurement_type` 必须为 `"proxy"`，且不套用端侧硬阈值（如 ≤10 W、<5 min 降频），只观察**趋势**（功耗档 → 持续吞吐衰减）。

### 2.2 被测模型注册表

| id | 模型 | 范式分组 | 框架 | 任务类型 | 训练数据要求 | 默认输入 | 权重/来源 | 备注 |
|---|---|---|---|---|---|---|---|---|
| `m.yolov8n` | YOLOv8n | conv（O(n) 基线） | ultralytics | 监督检测 | 需框标注 | 640 | fabric-defect-detection (.pt) | 最轻 |
| `m.yolov8s` | YOLOv8s | conv | ultralytics | 监督检测 | 需框标注 | 640 | 同上 | |
| `m.yolov11n` | YOLOv11n | conv | ultralytics | 监督检测 | 需框标注 | 640 | 同上 | |
| `m.patchcore` | PatchCore | feature-embed | anomalib / 本地 repo | 异常检测 | 仅正常样本 | 256/224 | anomaly_detection_for_textile_industry | 含 memory bank，采样比 `coreset_sampling_ratio` |
| `m.rd4ad` | RD4AD | feature-embed | 同上 | 异常检测 | 仅正常样本 | 256 | 同上 | 师生蒸馏 |
| `m.efficientad` | EfficientAD (S/M/L) | feature-embed | 同上 | 异常检测 | 仅正常样本 | 256 | 同上 | 三档规模 |
| `m.moeclip` | MoECLIP | attention/多模态（O(n²) 组） | PyTorch | 零样本异常检测 | 零样本（不训练目标域） | 336 | 需 OpenCLIP ViT-L-14-336px | 8 专家 / Top-2 路由 |

> 代码生成提示：每个模型需实现统一接口 `load(cfg) -> model`、`preprocess(img) -> tensor`、`infer(tensor) -> output`、`export(fmt) -> path`（fmt ∈ {onnx, fp16, int8}）。异常检测与检测模型的 `output` 语义不同（异常图/分数 vs 框），下游按 `task_type` 分支处理。

### 2.3 数据集注册表

| id | 数据集 | 标注类型 | 分辨率 | 规模 | 缺陷类别 | 有掩膜 | 训练集含异常 | 适配任务 |
|---|---|---|---|---|---|---|---|---|
| `d.mvtec` | MVTec AD | mask | 1024² | 5354，15 类 | 多种 | 是 | 否（仅正常） | 异常检测、量化分层 |
| `d.visa` | VisA | tag+mask | 多分辨率 | 10821（9621 正常/1200 异常），12 子集 | 表面+结构 | 是 | 否 | 异常检测、扫频 |
| `d.sdust` | SDUST-FDD | YOLO 框 | 512² | 训练 1650 / 测试 465 | 6 类（布匹） | 否 | 是 | YOLO 检测、并发 |
| `d.rawfabrid` | RAW-FABRID | COCO+mask | 1792×1024 灰度，可裁 256² | 训练 505 图→14196 正常块；测试 4969 正常/687 异常 | 未分类（布匹） | 是 | 否 | 异常检测、量化分层 |
| `d.zju` | ZJU-Leaper | tag+mask | 256² | 98777，8:2 分割（27650 异常/71127 正常） | 19 种纹理（织物） | 是 | 是 | 跨域漂移、换产、扫频 |

**关键属性（供代码判断合法性）：** `format ∈ {yolo, mask, coco}`；`has_mask ∈ {true,false}`（决定能否做缺陷尺寸分层）；`train_normal_only ∈ {true,false}`（决定能否训练异常检测模型）；`domains`（如 `d.zju` 的 19 种纹理，用于构造跨域划分）。

### 2.4 实验 × 数据集 × 模型 兼容矩阵

| 测试单元 | 推荐数据集 | 适用模型 | 关键前置/约束 |
|---|---|---|---|
| C1 功耗-热 | 任一（固定分辨率图流） | 全部 | 端侧或 proxy；30 min 连续 |
| C2 分辨率扫频 | `d.mvtec` / `d.visa` | 注意力组 `m.moeclip` vs 卷积组 `m.yolov8n,m.patchcore` | 分辨率梯度 256/512/1024/2048 |
| C3 并发 | `d.sdust` / `d.mvtec` | 全部 | 多实例，记录每路延迟 |
| C4 换产训练 | `d.zju`（按纹理型划分"旧型/新型"） | `m.yolov8n`、`m.patchcore/rd4ad/efficientad` | 全量训到收敛；增量微调 |
| C5 跨域漂移 | `d.zju` 纹理 / `d.visa` 类目（天然多源） | `m.patchcore/rd4ad/efficientad` | 域A训→域B测；禁用纯增广冒充 |
| C6 协同升级 | 任一（比一致性） | 任一部署模型 | 需两版推理引擎/权重/后处理 |
| M1 带宽 | `d.mvtec`（256 vs 2048） | 全部 | 端到端含 H2D/D2H |
| M2 量化 | 含掩膜：`d.mvtec/d.visa/d.rawfabrid/d.zju` | `m.yolov8n/8s`、`m.patchcore`、`m.efficientad` | 需像素级 GT 做尺寸分层 |
| N1 同步 | 不限（比权重体积/传输） | 全部（比模型大小） | tc 限速网络 |
| N2 回传 | 含缺陷视频流（`d.sdust` 帧序列拼流） | 检测类 | 注入 1% 缺陷帧 |
| N3 硬实时 | HIL（需 PLC/总线硬件） | 部署后模型 | 无硬件时列为"待验证" |

### 2.5 统一测量规范

所有涉及时间/吞吐的测量遵循同一规程，代码应封装为公共工具 `bench_utils`：

- **GPU 计时**：用 `torch.cuda.Event(enable_timing=True)` 成对打点，测量前后各 `torch.cuda.synchronize()`；禁止用 `time.time()` 直接测 GPU 段。
- **Warmup / 重复**：每配置 `warmup=100` 次 + `iters=1000` 次正式测量（训练类除外）。
- **重复实验**：每个测试整体重复 `repeats=5` 次（不同随机种子/不同进程启动），取均值 ± 标准差。
- **分位数**：延迟报告 `mean, std, p50, p95, p99, max`。
- **置信区间**：跨架构/跨分辨率对比给出 **bootstrap 95% CI**（`n_boot=1000`）；两组差异跨越 CI 才计为"有效差异"。
- **固定环境**：锁定 batch、分辨率、精度、CUDA/驱动版本；测量期间关闭其他 GPU 负载。
- **训练终止口径**：C4 的"全量训练"以**收敛准则**为终止（早停 patience，或验证指标平台期），记录真实耗时；"smoke test"（如 3 epoch）只用于跑通/测吞吐，二者分开记录、不混用。

### 2.6 统一结果 Schema（JSON）

每个测试单元运行后输出一个 `result.json`，顶层字段固定；`metrics` 内容按单元定义。字段名即代码里的 key。

```json
{
  "test_id": "C2",
  "test_name": "resolution_throughput_sweep",
  "timestamp": "2026-07-07T10:00:00Z",
  "measurement_type": "real",            // real | proxy
  "hardware_id": "hw.cloud",
  "env": {
    "gpu": "RTX 5090", "driver": "555.xx", "cuda": "12.4",
    "cpu": "...", "ram_gb": 128,
    "torch": "2.3.0", "framework": "ultralytics 8.x", "os": "Ubuntu 22.04"
  },
  "config": {                            // 本次运行的全部可变参数（用于复现）
    "model_id": "m.moeclip", "dataset_id": "d.mvtec",
    "precision": "fp32", "batch_size": 1,
    "resolutions": [256,512,1024,2048], "warmup": 100, "iters": 1000, "repeats": 5
  },
  "metrics": { "...": "见各测试单元的输出契约" },
  "hypothesis": { "id": "H2", "verdict": "unfilled", "evidence": "" }, // supported|partial|not_supported|unfilled
  "artifacts": ["plots/C2_curve.png", "raw/C2_latency.csv"],
  "notes": ""
}
```

**元数据（env）为强制字段**——没有它结果不可复现。建议由公共函数 `collect_env()` 自动采集。

### 2.7 建议的代码目录结构

```
benchmark_suite/
├── configs/
│   ├── registry.yaml          # §2.2–2.4 模型/数据集/硬件/兼容矩阵
│   ├── measurement.yaml       # §2.5 warmup/iters/repeats/percentiles/ci
│   └── hypotheses.yaml        # §1.3 假设与判据（阈值占位）
├── common/
│   ├── bench_utils.py         # 计时、分位数、bootstrap CI、warmup 循环
│   ├── env.py                 # collect_env()
│   ├── registry.py            # 加载注册表，校验兼容性
│   ├── models.py              # 统一模型接口 load/preprocess/infer/export
│   ├── datasets.py            # 统一数据接口，has_mask/尺寸分层
│   ├── power.py               # pynvml / tegrastats 采样，功耗封顶
│   └── report.py              # 汇总 result.json → 表格+曲线+结论
├── compute/  C1..C6 各一个模块
├── memory/   M1, M2
├── communication/  N1, N2, N3
├── results/  <test_id>/<timestamp>/result.json + artifacts/
└── run.py    # 编排入口：run.py --tests C2,M2 --models m.moeclip,m.yolov8n
```

### 2.8 执行与验证模式（三档执行模型）

真实测量不在开发机上进行，代码按三档执行模型运行：

| 档 | 在哪 | 做什么 | 关键约定 |
|---|---|---|---|
| **本地 local** | 开发机（无 GPU/性能有限） | **仅可行性验证**：导入、装配、用合成输入走通 `load→preprocess→infer`，校验注册表与结果契约 | `role=feasibility`；`device` 回退 cpu；缺库/权重/数据时用 **mock** 占位；循环参数缩到 `iters=2` |
| **服务器 server** | CUDA GPU 服务器 | **真实测量**：全部基准与场景测试 | `role=measurement`；`device=cuda`；`allow_mock=false`（缺依赖即报错，不静默 mock） |
| **数据 cloud** | 云端存储 | 数据集与权重推送至云端，server 从云端加载 | `data_mode=cloud`；`dataset_root/weights_root` 在 `runtime.yaml` 配置 |

约定：本地 `python run.py --check` 应全部走通（backend=mock 属正常）；服务器 `--mode server` 用真实模型/GPU/云端数据。测量类结果（result.json）只在 server 产生；local 只回答"程序能否跑起来"。

---

## 3. 测试单元详述

> 统一模板字段：**目的 / 对应挑战·假设 / 输入契约 / 被测组合 / 参数 / 测量指标 / 工具与 API / 执行步骤 / 输出契约 / 判据 / 风险与代理**。
> "输入契约""输出契约"是代码生成的关键——它们定义了每个模块的函数签名与返回结构。

---

### 计算开销维度

#### C1 —— 端侧资源受限：持续负载功耗-热-降频测试

- **目的**：验证在功耗/散热受限（无风扇被动散热）下，模型能否维持产线节拍所需的持续实时性。
- **对应挑战 / 假设**：功耗墙与无风扇散热约束下的实时推理矛盾 / **H1（hard）**。
- **输入契约**：`model_id`；固定分辨率的图像流（可循环同一批图）；`duration_s=1800`（30 min）；采样频率 `sample_hz=1`；（proxy 时）`power_caps_w=[30,50,70]`。
- **被测组合**：全部模型 × `hw.edge`（实测）或 `hw.cloud`+功耗封顶（proxy）。
- **参数**：`duration_s, sample_hz, power_caps_w, resolution, batch_size=1`。
- **测量指标**：

  | 指标 | 字段名 | 测量方式 | 判据类型 |
  |---|---|---|---|
  | 单帧延迟(ms) | `latency_ms` | cuda.Event / perf_counter | — |
  | 持续吞吐(FPS) | `fps_sustained` | 30 min 平均 | hard：≥ 节拍 |
  | 峰值→稳态吞吐衰减 | `fps_decay_pct` | (峰值−稳态)/峰值 | prior |
  | 功耗(W) | `power_w` | pynvml / tegrastats / 外接功率计 | proxy 参考 |
  | 温度(°C) | `temp_c` | pynvml / tegrastats | prior：<85 |
  | 首次降频时间(s) | `throttle_onset_s` | 频率跌破基频时刻 | prior |

- **工具与 API**：`pynvml.nvmlDeviceGetPowerUsage`（mW）、`nvmlDeviceGetTemperature`、`nvmlDeviceGetClockInfo`；Jetson 用 `tegrastats --interval 1000` 或读 `/sys/bus/i2c/.../in_power0_input`（INA3221）；功耗封顶 `nvidia-smi -pl`。
- **执行步骤**：
  1. 设定（proxy）功耗档或（实测）环境温度；锁定分辨率与 batch。
  2. 启动后台采样线程（`sample_hz`）记录 power/temp/clock 时间序列。
  3. 连续推理 `duration_s`，记录每帧延迟。
  4. 输出时间序列 + 稳态统计；绘制"温度/功耗/FPS 随时间"曲线。
- **输出契约**：`metrics = { fps_peak, fps_sustained, fps_decay_pct, throttle_onset_s, power_w:{mean,max}, temp_c:{mean,max}, timeseries_csv }`。
- **判据**：`fps_sustained ≥ line_speed_fps`（实例 30）→ H1 不成立；否则 H1 成立。proxy 结果仅报趋势。
- **风险与代理**：桌面 GPU 无法体现无风扇降频，proxy 只能反映"功耗档→吞吐"关系；如需真实结论必须上 `hw.edge`。

#### C2 —— 注意力复杂度：分辨率-吞吐扫频测试

- **目的**：验证含注意力骨干的模型在分辨率升高时吞吐下降是否显著快于纯卷积模型（O(n²) vs O(n)）。
- **对应挑战 / 假设**：高分辨率实时吞吐率冲突 / **H2（prior）**。
- **输入契约**：`model_ids`（须显式分为 `attention_group=[m.moeclip]` 与 `conv_group=[m.yolov8n, m.patchcore]`）；`resolutions=[256,512,1024,2048]`；`dataset_id`。
- **参数**：`resolutions, warmup=100, iters=1000, repeats=5, batch_size=1, precision=fp32`。
- **测量指标**：

  | 指标 | 字段名 | 测量方式 |
  |---|---|---|
  | 理论计算量(GFLOPs) | `gflops` | thop/fvcore/ptflops |
  | 实测吞吐(FPS) | `fps` | 各分辨率下 iters 次均值 |
  | 吞吐-分辨率曲线斜率 | `slope` | log-log 线性拟合斜率 |
  | 各阶段耗时占比 | `stage_breakdown` | torch.profiler（backbone/attn/head） |
  | 显存带宽利用率 | `mem_bw_util` | nvidia-smi dmon -s m |

- **工具与 API**：`thop.profile` 或 `fvcore.nn.FlopCountAnalysis`（理论 FLOPs）；`torch.profiler.profile(activities=[CPU,CUDA], with_flops=True)` 做分层耗时；`torch.cuda.Event` 计时。
- **执行步骤**：
  1. 对每个模型、每个分辨率，跑 warmup+iters，记录 FPS 与显存/带宽。
  2. 对每组做 log(FPS)–log(res) 线性拟合，取斜率 `slope`。
  3. 比较 `attention_group` 与 `conv_group` 的斜率分布（bootstrap CI）。
  4. 绘制"分辨率-吞吐"双组曲线。
- **输出契约**：`metrics = { per_model: { res: {fps, gflops, mem_bw_util, stage_breakdown} }, slope_by_group:{attention:{mean,ci}, conv:{mean,ci}}, slope_gap:{value,ci} }`。
- **判据**：`slope_gap.ci` 不跨 0 且注意力组更陡 → H2 支持；否则不支持/部分支持。
- **风险**：MoECLIP 在 2048 可能 OOM——需记录 OOM 分辨率本身即为一种证据；必要时降 batch 或分块。

#### C3 —— 扩展性瓶颈：多实例并发压力测试

- **目的**：验证单卡承载的并发检测路数上限，量化"每增一路的延迟代价"，对比不同范式的并发成本。
- **对应挑战 / 假设**：多工位并发部署成本与工程复杂度 / **H4（prior）**。
- **输入契约**：`model_id`；`concurrency=[1,2,4,8,16,32]`；每实例独立输入流；`latency_budget_ms`（实例 33）。
- **参数**：`concurrency 列表, per_stream_fps_target, duration_s, isolation ∈ {process, cuda_stream}`。
- **测量指标**：`per_stream_latency_ms`（各分位）、`total_fps`、`gpu_util`、`gpu_mem_mb`、`cpu_util`、`max_streams_under_budget`（延迟≤预算时的最大路数）、`cost_per_stream`（硬件成本/最大路数）。
- **工具与 API**：`multiprocessing` 或多 `torch.cuda.Stream`；`pynvml` 采 util/mem；`psutil` 采 CPU；可选 Triton Inference Server 做工业级并发对照。
- **执行步骤**：对每个并发数 N 启动 N 个推理实例 → 稳态后统计每路延迟与总吞吐 → 找"延迟翻倍点"和"吞吐饱和点" → 绘制并发-延迟、并发-吞吐曲线。
- **输出契约**：`metrics = { per_N: {latency_p50,p95,p99, total_fps, gpu_util, gpu_mem_mb, cpu_util}, max_streams_under_budget, saturation_N }`。
- **判据**：对比监督组与异常检测组的 `max_streams_under_budget`；异常检测（尤其 PatchCore memory bank）显著更低 → H4 支持。
- **风险**：预处理可能成 CPU 瓶颈——须同时记录 `cpu_util` 以免误判为 GPU 受限。

#### C4 —— 训练硬件门槛：换产训练成本测试

- **目的**：量化"换产"时重训/增量的时间与资源代价，验证边缘能否独立完成换产适配。
- **对应挑战 / 假设**：换产引发云端重训延迟与边缘增量学习局限 / **H3（mixed）**。
- **输入契约**：基础数据集 + 3 个"新型号"子集（实例：`d.zju` 按纹理型划出，样本量 50/200/1000）；训练场景枚举见下。
- **场景枚举**：`A_full`（全新类别从零训到收敛）、`B_incremental`（预训练+10% 新数据微调）、`C_fewshot`（5–20 张适配）、`D_edge_incremental`（在 `hw.edge` 上增量；无设备则标 `待验证`）。
- **参数**：`convergence: {metric, patience, max_epochs}`、`new_sample_sizes=[50,200,1000]`、`hardware_id`。
- **测量指标**：`train_time_full_h`、`train_time_incremental_min`、`energy_kwh`（功率×时间）、`gpu_mem_peak_gb`、`final_metric`（mAP 或 AUROC）、`edge_vs_cloud_ratio`。
- **工具与 API**：训练用各模型原生入口（ultralytics `train`；`main.py --baseline ... --run-transfer-learning`）；能耗 `pynvml` 积分功率×时间；`torch.cuda.max_memory_allocated`。
- **执行步骤**：分别在 `hw.cloud` 与（若有）`hw.edge` 跑 A/B/C/D → 记录耗时/能耗/峰值显存/最终精度 → 对比 全量 vs 增量 时间差、云 vs 边效率比、样本量-性能边际曲线。
- **输出契约**：`metrics = { per_scenario_per_hw: {train_time, energy_kwh, gpu_mem_peak_gb, final_metric}, full_vs_incremental_speedup, edge_vs_cloud_ratio, sample_marginal_curve }`。
- **判据**：全量训练耗时（小时级）与边/云效率比佐证 H3；D 场景不可行本身即为"边缘局限"的证据。
- **风险**：**务必以收敛为终止**，不可用 3 epoch 冒充（见 §2.5），否则耗时/精度失真。

#### C5 —— 数据分布异构性：跨域泛化与漂移测试

- **目的**：量化跨工位（跨域）的性能退化，检验中心统一模型 vs 独立模型 vs 聚合模型。
- **对应挑战 / 假设**：跨机台概念漂移与中心聚合退化 / **H5（prior）**。
- **输入契约**：**天然多源**域划分（实例：`d.zju` 的不同纹理型作为不同"工位"；或 `d.visa` 不同类目）；禁止仅用光照/噪声增广冒充真实漂移（增广仅作补充扰动并标注 `synthetic_shift=true`）。
- **训练方案**：`center`（合并训练）、`per_domain`（各域独立）、`federated`（**仅单轮 FedAvg 方向性探针**，或整体移出留待专项）。
- **参数**：`domains 列表, drift_schedule（逐步引入新纹理/新缺陷）`。
- **测量指标**：`in_domain_metric`、`cross_domain_metric`、`degradation_pct`、`federated_vs_percross`（聚合是否优于单域）、`drift_detection_latency`（分布变化→检出漂移的时长）。
- **工具与 API**：域划分与评估用 `datasets.py`；漂移检测可用特征分布距离（如 MMD / KS 检验）实现 `drift_detection_latency`。
- **执行步骤**：域A训→域A/域B测 → 记录 in/cross 指标与退化幅度 → （可选）单轮 FedAvg 聚合后测全局精度 → 模拟渐进漂移，画性能衰减曲线。
- **输出契约**：`metrics = { in_domain, cross_domain, degradation_pct:{value,ci}, federated_probe?, drift_curve }`。
- **判据**：`degradation_pct` 显著为正（跨 CI）→ H5 支持。
- **风险**：联邦学习易范围蔓延，务必限定为单轮探针；`synthetic_shift` 数据不得单独作为跨工位结论。

#### C6 —— 生命周期运维：协同升级一致性测试（新增，补齐表4 第6条）

- **目的**：验证"相机固件/推理引擎/模型权重/后处理逻辑"分层升级时，是否引发判定阈值漂移与检测不一致。
- **对应挑战 / 假设**：固件、模型与检测逻辑协同升级复杂性 / **H6（mixed）**。
- **输入契约**：同一模型的两个版本组件（`engine_v1/v2`、`weights_v1/v2`、`postproc_v1/v2`）；固定标定集 `calib_set`（同一批标注图，用于对比结论）。
- **参数**：`upgrade_axes ∈ {engine, weights, postproc}`、`canary_ratio`、`rollback=true`。
- **测量指标**：`threshold_drift`（升级后最优判定阈值变化）、`decision_disagreement_rate`（同一标定集，升级前后结论不一致比例）、`canary_time_s`、`rollback_time_s`、`output_distribution_shift`（异常分数分布 KL/KS）。
- **工具与 API**：多版本推理引擎（如 ONNXRuntime vs TensorRT，或不同版本）；阈值重标定脚本；分布距离用 scipy。
- **执行步骤**：固定标定集 → 分别用 v1、v2 组件推理 → 比较分数分布、最优阈值、逐样本结论 → 记录灰度/回滚耗时。
- **输出契约**：`metrics = { per_axis: {threshold_drift, decision_disagreement_rate, output_distribution_shift}, canary_time_s, rollback_time_s }`。
- **判据**：`decision_disagreement_rate` 与 `threshold_drift` 显著 → H6 支持（说明升级需重标定，运维成本高）。
- **风险**：需准备两套组件；若暂无第二版本，可用"精度导出差异"（FP32 vs FP16 引擎）作为最小可行版本。

---

### 内存开销维度

#### M1 —— HBM 带宽瓶颈：带宽竞争与数据搬运测试

- **目的**：验证高速采集写入与推理读取共享总线时，内存带宽是否封顶有效吞吐、数据搬运占比是否偏高。
- **对应挑战 / 假设**：图像高速采集与推理对总线带宽的竞争 / **H7（prior）**。
- **输入契约**：`model_id`；两种数据通路：`resident`（数据已在显存）与 `e2e`（磁盘/相机→CPU→GPU→推理）；对比变量见下。
- **对照实验**：`256px vs 2048px`（分辨率）、`batch 1 vs 8`、`统一内存 vs 显式拷贝`（Jetson 或 proxy）。
- **参数**：`resolutions, batch_sizes, io_source ∈ {disk, synthetic_camera}`。
- **测量指标**：`gpu_mem_mb`（峰值）、`mem_bw_gbs`、`h2d_ms/d2h_ms`（拷贝耗时）、`copy_ratio`（搬运/端到端）、`throughput_drop_pct`（采集+推理并发 vs 纯推理）。
- **工具与 API**：`nvidia-smi dmon -s mu` 或 Nsight/DCGM 采带宽；`torch.profiler` 分离 H2D/D2H；`torch.cuda.max_memory_allocated`；合成高速采集：独立线程按 `4K@60fps` 写入 pinned memory。
- **执行步骤**：先测 `resident` 基线吞吐 → 再测 `e2e` 全链路吞吐 → 差值即数据搬运开销 → 扫 256↔2048、batch 1↔8 → 记录带宽利用率是否饱和。
- **输出契约**：`metrics = { per_config: {gpu_mem_mb, mem_bw_gbs, h2d_ms, d2h_ms, copy_ratio, throughput_resident, throughput_e2e, throughput_drop_pct} }`。
- **判据**：高分辨率下 `mem_bw_util` 接近饱和且 `copy_ratio` 高 → H7 支持。
- **风险**：桌面 PCIe 与 Jetson 统一内存架构不同，"统一内存 vs 显式拷贝"对比在无 Jetson 时标 proxy。

#### M2 —— 存储层次约束：量化-微缺陷抹除测试

- **目的**：验证低位宽量化（INT8/INT4）是否抹除微弱缺陷特征，尤其对依赖特征距离的异常检测方法。
- **对应挑战 / 假设**：低位宽量化对微弱缺陷纹理特征的抹除 / **H8（mixed）**。
- **输入契约**：`model_id`（须含 `m.patchcore`/`m.efficientad` 与 `m.yolov8n/8s`）；含**像素掩膜**的数据集（`d.mvtec/d.visa/d.rawfabrid/d.zju`）以支持缺陷尺寸分层。
- **精度矩阵**：`FP32(基准) / FP16 / INT8(PTQ) / INT8(QAT) / INT4`。
- **量化策略对照**：`per_tensor / per_channel / mixed_precision / QAT`。
- **参数**：`calib_set_size=100..500`（正常图校准）、`size_bins=[(0,10),(10,30),(30,inf)]`（像素）。
- **测量指标**：`model_size_mb`、`speedup`、`overall_metric`（mAP/AUROC）、`recall_by_size`（分尺寸召回率）、`micro_recall_drop`（<10px 量化前后差）、`localization_iou_delta`。
- **工具与 API**：导出/量化用 TensorRT（INT8 calibrator）或 ONNX Runtime static quantization；QAT 用 `torch.ao.quantization`；尺寸分层由掩膜连通域面积计算。
- **执行步骤**：以 FP32 为基准 → 导出各精度 → 用 100–500 张正常图校准 INT8 → 测集评估 → **按缺陷尺寸分层统计召回率变化** → 对照四种量化策略对微缺陷的影响。
- **输出契约**：`metrics = { per_precision: {model_size_mb, speedup, overall_metric, recall_by_size, micro_recall_drop, localization_iou_delta}, strategy_comparison }`。
- **判据**：`micro_recall_drop`（<10px）显著大于大缺陷 → H8 支持。
- **风险**：YOLO（框标注 `d.sdust`）无掩膜，不能做像素分层——分层实验只在含掩膜数据集上跑。

---

### 通信开销维度

#### N1 —— 分布式版本同步：模型分发与同步测试

- **目的**：量化模型更新下发到多工位的耗时与带宽冲击，验证异常检测模型同步开销是否高于监督模型、版本一致性风险。
- **对应挑战 / 假设**：模型更新下发的高效性与多机台版本同步 / **H9（prior）**。
- **更新策略**：`full`（全量）、`delta`（权重增量）、`structured`（仅特定层，如检测头）、`distill`（蒸馏小模型）。
- **网络场景**：`lan(1Gbps 有线,10台)`、`factory(100Mbps 有丢包,50台)`、`edge_wan(10Mbps 无线高延迟,100台)`。
- **输入契约**：各模型权重文件（比体积）；模拟节点数 `N`；网络参数经 `tc` 设置。
- **参数**：`strategies, network_profiles, node_counts=[10,50,100]`。
- **测量指标**：`update_pkg_mb`（各策略）、`per_device_update_s`（下载+校验+加载）、`fleet_sync_s`（N 台全部完成，P50/P95/P99）、`peak_bw_mbps`、`version_consistency`（更新后版本分裂检测）。
- **工具与 API**：`tc qdisc ... netem delay/loss rate`（带宽/延迟/丢包）；本地起 N 个节点进程 + 1 中心服务器；打点下载/校验/加载各阶段时间。
- **执行步骤**：搭模拟网络 → 触发更新 → 记录各节点各阶段时间戳 → 统计 fleet 同步分布 → 对比各策略带宽/时间节省。
- **输出契约**：`metrics = { per_strategy_per_network: {update_pkg_mb, per_device_update_s, fleet_sync_s:{p50,p95,p99}, peak_bw_mbps}, version_consistency }`。
- **判据**：异常检测模型（含 memory bank，包体积大）`fleet_sync_s` 显著高于 YOLO → H9 支持。
- **风险**：PatchCore memory bank 使更新包远大于权重文件，须按实际部署形态计包体积。

#### N2 —— 非对称数据流控：事件驱动回传延迟测试

- **目的**：验证"仅缺陷事件才回传"能否大幅省带宽，同时保证高危缺陷毫秒级响应。
- **对应挑战 / 假设**：高危缺陷图像块的毫秒级异步回传 / **H10（hard）**。
- **回传方案**：`full`（全帧上传,基线）、`event`（检出才回传块）、`tiered`（高危→整图+位置、中危→裁剪区域、低置信→仅特征向量）。
- **输入契约**：测试视频流，注入 `defect_rate=1%` 缺陷帧（实例：`d.sdust` 帧序列拼流）；边缘端跑检测模型。
- **参数**：`defect_rate, scheme, duration_h=1`。
- **测量指标**（埋点 T1 检出→T2 编码→T3 发送→T4 云端接收→T5 入库）：`detect_to_arrival_ms`、`payload_kb_per_event`、`normal_frame_zero_upload_rate`、`concurrent_event_success_rate`（1s 内 10 个并发缺陷）、`bw_saving_pct`（vs full）、`false_alarm_extra_traffic`（异常检测误报导致的额外回传）。
- **工具与 API**：埋点用 `time.perf_counter_ns`；网络经 `tc`；编码用 JPEG/PNG，记录压缩后大小。
- **执行步骤**：构造 1% 缺陷视频流 → 边缘检测触发回传 → 记录 T1–T5 各段延迟分布 → 统计 1h 总回传量 vs 全量 → 记录误报额外流量。
- **输出契约**：`metrics = { per_scheme: {detect_to_arrival_ms:{p50,p95,p99}, payload_kb_per_event, bw_saving_pct, normal_frame_zero_upload_rate, false_alarm_extra_traffic} }`。
- **判据**：`bw_saving_pct` 高（预期 90–99%）且高危 `detect_to_arrival_ms` 达标 → H10 支持；注意异常检测高误报可能抬高实际回传量。
- **风险**：局域网 50–200ms、广域网 500–2000ms 为文献先验，仅对照。

#### N3 —— 硬实时控制链协同：端到端控制延迟测试

- **目的**：验证检测信号与执行机构（织机/产线）的硬实时协同能否满足确定性时延要求。
- **对应挑战 / 假设**：检测信号与织布机的硬实时协同 / **H11（hard）**。
- **需硬件在环（HIL）**：图像采集卡 → 边缘推理 → PLC 模拟器；**无 PLC/总线硬件时本单元标 `待验证`，不出实测结论**。
- **输入契约**：部署后模型；模拟控制信号通道（GPIO / Modbus / CAN）；时间同步（PTP）。
- **延迟分解**：采集(曝光+读出) / 推理(前处理+推理+后处理) / 决策(等级判定) / 通信(→PLC) / 执行(PLC 扫描+机构响应)。
- **参数**：`cycles=1000, os_variant ∈ {vanilla_linux, preempt_rt, hw_accel}`。
- **测量指标**：`e2e_latency_ms`（各分位/最大）、`jitter_us`（标准差/最大）、`deadline_miss_rate`、`preemption_capability`。
- **工具与 API**：`time.perf_counter_ns` 或硬件计时；PTP（linuxptp）同步；对比 vanilla vs PREEMPT_RT 内核 vs FPGA/ASIC。
- **执行步骤**：搭 HIL → 各节点打 μs 级时间戳 → 跑 1000 次检测-控制循环 → 统计延迟/抖动分布 → 对比三种系统方案。
- **输出契约**：`metrics = { per_os_variant: {e2e_latency_ms:{mean,p99,max}, jitter_us, deadline_miss_rate} }` 或 `status: "pending_hardware"`。
- **判据**：纯软件方案 `p99/max` 超硬实时预算或抖动大 → H11 支持。
- **风险**：模型推理时延本身不确定，需时间戳补偿；缺硬件时仅能做软件段延迟画像，须显式标注不完整。

---

## 4. 执行流程与依赖

推荐分阶段推进，前一阶段产物是后一阶段输入（可作为编排 DAG）：

- **阶段 0 环境与注册**：落成 `configs/registry.yaml`、`measurement.yaml`、`hypotheses.yaml`；实现 `common/*`；`collect_env()` 自检通过。
- **阶段 1 数据准备**：按兼容矩阵（§2.4）准备各数据集、分辨率梯度、`d.zju` 的"旧型/新型"与跨域划分、缺陷尺寸分层掩膜。
- **阶段 2 模型就绪**：各模型 `load/export`（FP32/FP16/INT8）跑通；异常检测模型完成正常样本训练；MoECLIP 载入 OpenCLIP 权重。
- **阶段 3 计算+内存基准**：C2、C3、M1、M2（可在 `hw.cloud` 全量执行）。
- **阶段 4 场景化**：C1（端侧或 proxy）、C4、C5、C6。
- **阶段 5 通信与分布式**：N1、N2；N3 视 HIL 硬件情况执行或标 `待验证`。
- **阶段 6 汇总**：`report.py` 聚合所有 `result.json` → 生成对比表、曲线、逐假设判定，保留 null 结果。

依赖要点：C4 依赖阶段 2 的训练；M2 依赖阶段 2 的量化导出与阶段 1 的掩膜；C1/C4/N3 的"实测"结论依赖 `hw.edge`/HIL 到位，否则走 proxy 或 `待验证`。

---

## 5. 汇总与报告模板

`report.py` 输出一张主表（每假设一行），字段与 §1.3 对齐，运行后由实测填充，**不预判**：

| 假设 | 关键量（实测） | 均值±std | 95% CI / 显著性 | measurement_type | 判定 |
|---|---|---|---|---|---|
| H1 | fps_sustained | … | … | real/proxy | 支持/部分/不支持 |
| H2 | slope_gap | … | … | real | … |
| … | … | … | … | … | … |
| H11 | e2e p99 / jitter | … | … | real/pending | … |

同时输出：各测试单元的曲线图（C2 分辨率-吞吐、C3 并发-延迟、C1 时序、M2 分尺寸召回等）与原始 CSV，路径登记在各 `result.json` 的 `artifacts`。

---

## 6. 附录

### 6.1 工具清单（按用途）

- **推理基准/计时**：`torch.cuda.Event`、`torch.utils.benchmark`、`trtexec --benchmark`
- **FLOPs/分层**：`thop`、`fvcore`、`ptflops`、`torch.profiler`
- **GPU 功耗/显存/带宽**：`pynvml`、`nvidia-smi dmon -s pucm`、`nvidia-smi -pl/-lgc`、DCGM、Nsight
- **系统/CPU**：`psutil`、`powertop`
- **边缘/端侧**：`tegrastats`、`jtop`(jetson-stats)、INA3221 sysfs
- **量化/导出**：TensorRT（INT8 calibrator）、ONNX Runtime static quant、`torch.ao.quantization`（QAT）
- **并发**：`multiprocessing`、`torch.cuda.Stream`、Triton Inference Server
- **网络模拟**：`tc` / `netem`（delay/loss/rate）
- **实时/同步**：`time.perf_counter_ns`、linuxptp（PTP）
- **分布距离/漂移**：scipy（KS）、MMD 实现

### 6.2 术语对照（通用 ↔ 布匹实例）

检测工位（验布机/机台）｜产品换型（换布种/换产）｜产线节拍实时约束（布匹 30 FPS）｜高分辨率长幅面输入（线阵相机长条图）｜缺陷/微弱缺陷（疵点/微缺陷）｜行业分级验收标准（四分制扣分）。

### 6.3 与旧版 handbook 的主要差异

适用范围：布匹专用 → 工业通用（布匹为实例）；结论：预期全部✅ → 假设登记允许证伪；阈值：猜测当门槛 → 区分 hard/prior；硬件：默认端侧存在 → 实测/proxy 显式区分；注意力测试：全模型混测 → 注意力组 vs 卷积基线；数据集：自由混用 → 兼容矩阵锁定；运维挑战：无协议 → 新增 C6；统计：仅均值±std → 加 bootstrap CI；且新增统一 result.json 契约与代码目录结构以支持自动化实现。

---

*本手册为验证设计文档，不含实测数据。所有"☐ 待填 / （占位）"处需在硬件条件与判据阈值锁定后，由实测结果填充。*
