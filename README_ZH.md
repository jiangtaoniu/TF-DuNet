# TF-DuNet: 用于时间序列预测的时频双分支网络

<p align="center">
  简体中文 | <a href="README.md">English</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg" alt="PyTorch Version">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

## 简介

这是 **TF-DuNet** 的官方 PyTorch 实现代码。我们的工作致力于解决时间序列预测中的一个核心挑战：同时对高频瞬态动态和准平稳的多尺度结构进行建模。TF-DuNet 引入了一种新颖的非对称双分支架构，显式地解耦并专注于处理这两种互补的信息流。

<p align="center">
  <img src="assets/figure1.png" width="800"/>
</p>
<p align="center">
  <i>TF-DuNet 的架构，包含高保真时间分支 (ASST) 和结构化时频分支 (Decomposition & MSDB)。</i>
</p>

## 模型架构

TF-DuNet 采用双分支架构，包含：

1.  **ASST 分支 (自适应序列时空模块):** 一个高保真的时间分支，利用自注意力机制捕获全局的时空依赖关系。它直接处理输入序列，以提取高频和瞬态的动态变化。
2.  **分解分支 (Decomposition Branch):** 一个结构化的时频图分支，将时间序列分解为趋势 (trend) 和季节性 (seasonal) 分量（通过 DFT 或滑动平均）。它跨尺度地处理这些分量，并使用基于动态图的多尺度时空依赖块 (MSDB) 来建模空间网络拓扑关系。

这两个特定分支的输出随后通过可配置的融合策略（如 add, gating, film, cross_attention 等）动态合并，从而生成最终的预测结果。

## 项目结构

```
.
├── dataset/            # 数据集目录 (需要手动下载数据存放于此)
├── models/             # 核心模型实现代码
│   └── TF_DuNet.py     # 主模型架构 (TF-DuNet)
├── layers/             # 自定义神经网络层和模块 (ASST, MSDB 等)
├── data_provider/      # 数据加载和预处理脚本
├── exp/                # 实验运行及训练代码 (train, validate, test)
├── utils/              # 辅助函数和评估指标工具
├── run_PEMS08.py       # 实验运行的入口脚本
├── requirements.txt    # Python 依赖包列表
├── README.md           # 英文项目文档
└── README_ZH.md        # 中文项目文档
```

## 安装指南

### 1. 配置环境
我们推荐使用 Conda 来管理 Python 环境。

```bash
conda create --name TF-DuNet python=3.8 -y
conda activate TF-DuNet
```

### 2. 安装依赖库

安装 PyTorch (请根据您的系统及 GPU 环境调整 CUDA 版本):
```bash
pip install torch==2.1.0+cu118 torchaudio==2.1.0+cu118 torchvision==0.16.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```

安装其他必须的 Python 包:
```bash
pip install -r requirements.txt
```

### 3. 准备数据
下载项目所需的数据集，并将其放置在 `./dataset/` 目录中。

期望的目录结构如下：
```
dataset/
├── PEMS/
│   └── PEMS08.npz
├── ETT/
│   └── ETTh1.csv
└── ...
```

## 使用方法

您可以使用提供的入口脚本 `run_PEMS08.py` 来训练和评估模型。

### 基础训练命令格式

```bash
python run_PEMS08.py --is_training 1 \
  --model_id PEMS08 \
  --model TF_DuNet \
  --data PEMS \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS08.npz \
  --seq_len 96 \
  --pred_len 12 \
  --d_model 128 \
  --e_layers 5 \
  --asst_e_layers 3 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --task_name long_term_forecast \
  --final_fusion_method add
```

### 核心参数说明 (基于 `run_PEMS08.py`)

- `--task_name`: 预测任务类型 (例如 `long_term_forecast` 进行长期预测)。
- `--seq_len`: 输入的历史序列长度。
- `--pred_len`: 目标预测序列长度。
- `--final_fusion_method`: ASST 分支和分解分支之间的最终融合方法 (可选值: `add`, `static_weighted`, `concat`, `gate`, `film`, `cross_attention`, `learnable_weighted`, `custom_ranged`)。
- `--msdb_internal_fusion_method`: MSDB 模块内部特征使用的融合方法 (可选值: `add`, `static_weighted`, `concat`, `gate`, `film`, `cross_attention`)。
- `--decomp_method`: 时间序列分解策略 (`dft_decomp` 代表傅里叶变换分解，或 `moving_avg` 滑动平均)。
- `--asst_e_layers`: ASST 分支中编码器层的数量。
- `--e_layers`: 分解分支中编码器层的数量。

## 引用

如果您在研究中使用了本项目的代码或思路，请考虑引用我们的论文：
```bibtex
@article{your_name2025tfdunet,
  title={TF-DuNet: A Time-Frequency Dual-Branch Network for Time Series Forecasting},
  author={Author, A. and Author, B.},
  journal={Journal or Conference},
  year={2025}
}
```

## 致谢
我们的实现参考了 [Time-Series-Library](https://github.com/thuml/Time-Series-Library) 的代码库。感谢原作者们的开源贡献。
