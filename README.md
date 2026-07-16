# SpecLoc

[![CI](https://github.com/DexZane/SpecLoc/actions/workflows/ci.yml/badge.svg)](https://github.com/DexZane/SpecLoc/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

SpecLoc 是一个可审计的遥感微小目标检测与局部频谱分析研究框架。核心问题是：

> 在控制目标尺寸、类别、局部对比度和背景复杂度后，局部频谱特征能否稳定增加对定位误差的解释力？

当前版本首先提供可复现的 RSOD 工程验收和 AI-TOD-v2 正式基线；它不声称已经得到
新的检测性能结论，也不包含未经 Gate 1 支持的复杂网络模块。

## 快速开始

### 1. 安装

RTX 4090 等 CUDA 12.1 服务器：

```bash
git clone https://github.com/DexZane/SpecLoc.git
cd SpecLoc
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

其他安装配置：

```bash
# CUDA 11.8
python -m pip install -r requirements/cuda118.txt
python -m pip install -e .

# 仅 CPU（开发与 CI）
python -m pip install -r requirements/cpu.txt
python -m pip install -e .

# 测试和 lint 工具（先完成任一运行环境安装）
python -m pip install -r requirements/dev.txt
```

所有配置都使用预编译 MMCV wheel；如果没有匹配的 wheel，安装会立即失败，不会悄悄
进入耗时的源码编译。核心实验版本固定在 `requirements/` 中，包括
`setuptools==80.10.2`、PyTorch 2.1.2、MMCV 2.1.0 和 MMDetection 3.3.0。

### 2. 放置 RSOD

将完整数据放到 `data/RSOD/`：

```text
data/RSOD/
├── images/
├── annotations/
│   ├── instances_train.json
│   ├── instances_val.json
│   ├── instances_test.json
│   └── instances_all.json
├── splits/
├── images.sha256
└── manifest.json
```

数据目录已被 Git 忽略，不会被提交到代码仓库。

### 3. 检查、训练和评估

```bash
specloc info
specloc doctor rsod
specloc train rsod
specloc evaluate rsod
```

默认训练合同为：单卡、AMP、每卡 batch size 2、梯度累积 2 步。RTX 4090 无需再写
MMDetection 的长串 `--cfg-options`。

中断后恢复：

```bash
specloc train rsod --resume
```

常用覆盖：

```bash
specloc train rsod --batch-size 4
specloc train rsod --work-dir work_dirs/rsod_trial_001
specloc evaluate rsod --checkpoint /path/to/model.pth
```

查看所有参数：

```bash
specloc --help
specloc train --help
```

## 研究路线

1. **RSOD / Gate -1**：跑通安装、数据校验、训练、验证、测试和导出；
2. **AI-TOD-v2 / Gate 0**：建立可信的微小目标标准基线；
3. **频谱信息量 / Gate 1**：比较 Size-control 与 Size+Spectrum；
4. 只有 Gate 1 通过后，才开发有界的频谱残差定位机制。

AI-TOD-v2 使用示例：

```bash
export AITOD_SOURCE='<实际授权来源>'
export AITOD_LICENSE_NOTE='<已核实的许可或访问条款>'
specloc doctor aitod
specloc train aitod
specloc evaluate aitod
```

## 项目结构

```text
SpecLoc/
├── .github/                 GitHub Actions CI
├── configs/                 MMDetection 实验配置
├── data/                    本地数据挂载点（不提交数据）
├── docs/                    实验和复现文档
├── requirements/            CPU/CUDA/开发依赖锁定文件
├── scripts/                 数据准备、校验和兼容脚本
├── src/specloc/             可安装 Python 包与统一 CLI
├── tests/                   单元和合同测试
├── tools/                   MMEngine 底层训练、测试和分析入口
├── pyproject.toml           包元数据与工具配置
└── requirements.txt         默认 CUDA 12.1 一步安装入口
```

## 开发

```bash
python -m pip install -r requirements/dev.txt
python -m pytest tests/ -q
python -m ruff check src tests scripts tools
```

底层兼容入口 `tools/train.py`、`tools/test.py` 和
`scripts/runCloudExperiment.sh` 继续保留，但新用户优先使用 `specloc` 命令。

数据约定见 [data/README.md](data/README.md)，依赖配置说明见
[requirements/README.md](requirements/README.md)。个人运行手册属于本地资料，不上传仓库。

## 证据边界

- RSOD 只用于工程流程验收，不承担论文机制结论；
- AI-TOD-v2 manifest 未验证时，正式训练会停止；
- Gate 1 返回 `stop` 是有效科学结果，不是程序失败；
- 每次正式运行记录 Git commit、依赖、数据 manifest、随机种子和硬件信息。

## 许可与公开范围

SpecLoc 自有代码使用 [Apache License 2.0](LICENSE)。部分命令行工具和评估常量
改编自 OpenMMLab 与 cocoapi-aitod，版权与许可说明见
[第三方许可说明](docs/THIRD_PARTY_NOTICES.md)。

本仓库不分发 RSOD、AI-TOD/AI-TOD-v2 原始数据、派生标注、预训练权重、训练权重或
实验输出。数据集与权重不适用 SpecLoc 的 Apache-2.0 许可，必须分别遵守上游条款。
