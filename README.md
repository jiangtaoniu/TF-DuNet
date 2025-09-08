# TF-DuNet: A Time-Frequency Dual-Branch Network for Time Series Forecasting

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg" alt="PyTorch Version">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

This is the official PyTorch implementation of **TF-DuNet**. Our work tackles a core challenge in time series forecasting: simultaneously modeling high-frequency, transient dynamics and quasi-stationary, multi-scale structures. TF-DuNet introduces a novel asymmetric dual-branch architecture to explicitly decouple and specialize in processing these two complementary information streams, achieving state-of-the-art performance on numerous benchmarks.

<p align="center">
  <img src="assets/figure1.png" width="800"/>
</p>
<p align="center">
  <i>The architecture of TF-DuNet, featuring a high-fidelity temporal branch (ASST) and a structural time-frequency branch (GSSTM).</i>
</p>

## Getting Started

### 1. Setup Environment
```bash

conda create --name TF-DuNet python=3.8 -y
```

```bash
conda activate TF-DuNet
```

```bash
pip install torch==2.1.0+cu118 torchaudio==2.1.0+cu118 torchvision==0.16.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```

```bash
pip install axial-positional-embedding==0.2.1 certifi==2022.12.7 charset-normalizer==2.1.1 colorama==0.4.6 cycler==0.12.1 einops==0.4.1 filelock==3.13.1 fsspec==2024.6.1 idna==3.4 jinja2==3.1.4 joblib==1.4.2 kiwisolver==1.4.7 local-attention==1.4.4 markupsafe==2.1.5 matplotlib==3.4.3 mpmath==1.3.0 networkx==3.0 numpy==1.22.4 packaging==25.0 pandas==1.1.5 patool==1.12 patsy==1.0.1 pillow==10.2.0 product-key-memory==0.1.10 pyparsing==3.1.4 python-dateutil==2.9.0.post0 pytz==2025.2 reformer-pytorch==1.4.4 requests==2.28.1 scikit-learn==1.2.1 scipy==1.8.0 six==1.17.0 sktime==0.4.1 statsmodels==0.14.1 sympy==1.11.1 threadpoolctl==3.5.0 tqdm==4.64.0 typing-extensions==4.12.2 urllib3==1.26.13

```

### 2. Prepare Data
Download the datasets from [DataSet](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2). Place the data files (`.csv` or `.npz`) into the `./dataset/` directory according to their names. The expected structure is:
```
dataset/
├── PEMS/
├───── PEMS04.npz
├── ETT/
├───── ETTh1.csv
└── ...
```

## Citation

If you find this work useful for your research, please consider citing our paper:
```bibtex
@article{your_name2025tfdunet,
  title={TF-DuNet: A Time-Frequency Dual-Branch Network for Time Series Forecasting},
  author={Author, A. and Author, B.},
  journal={Journal or Conference},
  year={2025}
}
```

## 致谢 (Acknowledgements)
Our implementation references the code base of [Time-Series-Library](https://github.com/thuml/Time-Series-Library). We thank the original authors for their open-sourcing.
