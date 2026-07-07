# benchmark_suite —— 开销挑战验证框架

对应 **Verification Handbook v3**。分步搭建。当前已完成 **核心脚手架 + 执行模型 + 模型适配层骨架**。

## 执行模型（§2.8，三档）

| 档 | 在哪 | 做什么 |
|---|---|---|
| `--mode local`（默认） | 开发机，无 GPU | **仅可行性验证**：装配链路走通，缺库/权重用 mock 占位 |
| `--mode server` | CUDA GPU 服务器 | **真实测量**，禁止 mock（缺依赖报错） |
| 数据 cloud | 云端 | 数据集/权重推云端，server 加载 |

真实 result.json 只在 server 产生；本地只回答"程序能不能跑起来"。

## 当前已搭建（阶段0）

```
benchmark_suite/
├── configs/
│   ├── registry.yaml       # §2.2-2.4 硬件/模型/数据集/兼容矩阵（唯一 id 来源）
│   ├── measurement.yaml    # §2.5 warmup/iters/repeats/percentiles/bootstrap
│   ├── hypotheses.yaml     # §1.3 假设 H1-H11 + 判据类型 + 阈值占位
│   └── runtime.yaml        # §2.8 执行档 local/server
├── common/
│   ├── env.py              # collect_env() 环境元数据（torch/pynvml 可选）
│   ├── bench_utils.py      # Timer / run_timed / summarize / bootstrap_ci
│   ├── registry.py         # 注册表加载 + 兼容性校验
│   ├── runtime.py          # 执行档解析：设备/mock/合成输入
│   ├── models.py           # 模型适配层：统一接口 + mock 回退 + 工厂 + 可行性检查
│   └── result.py           # 统一 result.json 契约的构建与落盘
├── run.py                  # 编排入口（--env/--validate/--list/--status/--check/--demo-result）
├── requirements.txt
└── results/                # 运行产物 <test_id>/<timestamp>/result.json
```

## 快速自测（无需 GPU/torch）

```bash
pip install pyyaml numpy
cd benchmark_suite
python run.py --validate      # 校验注册表一致性
python run.py --list          # 列出模型/数据集/测试单元
python run.py --status        # 各测试单元实现状态
python run.py --env           # 环境元数据
python run.py --demo-result   # 走通 result.json 契约
```

## 已搭建的适配层（当前步）

- `common/models.py` —— 统一接口 + `MockModel` 回退 + `get_adapter()` 工厂 + `feasibility_check()`
  - `UltralyticsAdapter`（YOLO）：真实载入路径已写（需 ultralytics + 权重）
  - `AnomalyRepoAdapter` / `MoECLIPAdapter`：接口就位，真实对接标 TODO，本地走 mock

- `common/datasets.py` —— 统一 `Sample` 流 + 合成回退 + 真实目录扫描；
  `defect_size_bins()`（M2 尺寸分层，scipy 连通域）、`split_by_domain()`（C5 跨域划分）

验证：`python run.py --check` 本地 7 模型 + 5 数据集全部走通（synthetic/mock）。

## 已实现的测试单元（11/11，local 可行性全部跑通）

计算 **C1–C6**、内存 **M1/M2**、通信 **N1/N2/N3** 全部实现。运行任意单元：
`python run.py --test <ID>`（如 `--test N1`）。所有单元核心判定逻辑均有单测。

**报告层已完成**：`python run.py --report` 聚合全部单元 → 逐假设判定总表（按维度分组+计数）
并写 `results/report.md`。`--source memory`（默认，内存跑）| `--source disk`（读 server 落盘结果）。

**兼容性解析器已完成**（`common/compat.py`）：把"模型↔数据集↔测试"的合法性规则写成代码。
`python run.py --compat [--test <ID>]` 列出每个测试的合法(模型×数据集)组合；
跑测试时若显式指定不兼容的 model+dataset 会被挡下（附原因）。规则见 compat.py 顶部注释。

剩余：各单元 server 真实路径对接（真实模型/量化/训练/网络/HIL）。

### 代表性结论（local 合成演示）

- **C2 分辨率-吞吐扫频**（`compute/c2_resolution_sweep.py`，假设 H2）：分辨率梯度 × log-log 斜率 ×
  注意力组 vs 卷积组对比。核心逻辑 `loglog_slope` / `judge_h2` 可单测。
  - 运行：`python run.py --test C2 --dataset d.mvtec`
  - local 用复杂度模型（注意力 O(res⁴) vs 卷积 O(res²)）演示 H2 supported；server 走真实计时
- **M2 量化-微缺陷抹除**（`memory/m2_quantization.py`，假设 H8）：精度矩阵
  FP32/FP16/INT8(PTQ/QAT)/INT4 × 按缺陷尺寸分层召回率 × micro_recall_drop × H8 判定。
  - 可单测核心逻辑：`recall_by_size` / `micro_recall_drop` / `judge_h8`
  - 运行：`python run.py --test M2 --model m.patchcore --dataset d.mvtec`
  - local 演示微缺陷召回随量化下降最多（判定 supported）；server 走真实量化评估

## 尚未实现（后续分步授权）

- M2 的 server 真实路径：`adapter.export/infer_quantized`（TensorRT/ONNXRuntime/torch.ao）
- `AnomalyRepoAdapter` / `MoECLIPAdapter` 的真实载入对接（需仓库上 PYTHONPATH + 权重/config）
- 各数据集特殊目录布局的真实扫描细化（现为通用 glob + 掩膜命名配对）
- `common/power.py` —— pynvml/tegrastats 采样、功耗封顶
- `common/report.py` —— result.json 聚合成表/曲线/逐假设判定
- `compute/` `memory/` `communication/` —— 11 个测试单元 C1-C6 / M1-M2 / N1-N3

## 设计约定（摘自 Handbook）

- 一切实体先在 `registry.yaml` 注册、后按 id 引用，不硬编码。
- 结果以假设登记形式记录，`verdict` 默认 `unfilled`，允许证伪。
- 区分 `measurement_type: real | proxy`；缺边缘硬件时用 GPU 代理并标注。
- 第三方依赖（torch/pynvml/psutil）均可选，缺失时降级，保证脚手架在纯 CPU 环境可跑。
