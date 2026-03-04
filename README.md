# SubspaceAD

<p align="center">
  <a href="https://huggingface.co/spaces/MoxVox/SubspaceAD-visual-demo" target="_blank">
    <img src="https://img.shields.io/badge/%20Live%20Demo-SubspaceAD-blue?style=for-the-badge" alt="SubspaceAD Demo">
  </a>
</p>

This repository contains the official implementation of the paper:

> **SubspaceAD: Training-Free Few-Shot Anomaly Detection via Subspace Modeling [CVPR2026]**

---

<p align="center"> <img src="assets/subspacead_overview.png" width="85%"> </p>

Figure 1. SubspaceAD consists of two training-free stages: (1) extracting DINOv2 patch features from a few normal exemplars, and (2) estimating a low-dimensional PCA subspace to model normal appearance. Anomalies are detected via reconstruction residuals.

## Introduction

Detecting visual anomalies in industrial inspection often requires operating with only a few normal exemplars per category. While many recent approaches rely on large memory banks, auxiliary datasets, or multi-modal tuning, **SubspaceAD** introduces a minimalist, training-free alternative.

SubspaceAD consists of two stages:

1. **Feature Extraction:** Patch-level features are extracted from a small set of normal images using a frozen **DINOv2** backbone.
2. **Subspace Modeling:** A **PCA** model is fit to these features to estimate the low-dimensional manifold of normal appearance.

At inference time, anomalies are detected using the reconstruction residual with respect to this learned subspace.
Despite its simplicity, SubspaceAD achieves **state-of-the-art** performance in one-shot and few-shot settings.

**Main Results (1-Shot)**

* **MVTec-AD:** 98.0% Image AUROC; 97.6% Pixel AUROC
* **VisA:** 93.3% Image AUROC; 98.3% Pixel AUROC

---

<p align="center">
  <table>
    <tr>
      <td align="center">
        <img src="assets/mvtec_examples.png" height="250">
        <br>
        <sub><b>MVTec-AD</b></sub>
      </td>
      <td align="center">
        <img src="assets/visa_examples.png" height="250">
        <br>
        <sub><b>VisA </b></sub>
      </td>
    </tr>
  </table>
</p>

<p align="left"><i>Figure 2. Qualitative comparison on VisA and MVTec-AD (1-shot). SubspaceAD produces sharper and more precise anomaly maps than
PromptAD and AnomalyDINO, with fewer false activations and better alignment with ground-truth defects across both datasets.</i></p>


## Environment Setup

```bash
# 1. Create environment
conda create -n subspacead python=3.10
conda activate subspacead

# 2. Install dependencies and the package
pip install -r requirements.txt
pip install -e .
```

## Data Preparation

### MVTec-AD

Download the dataset from the [MVTec website](https://www.mvtec.com/company/research/datasets/mvtec-ad) and extract it to:

```
datasets/mvtec-ad/
```

### VisA

```bash
# 1. Download and extract
mkdir -p datasets/VisA_20220922
wget https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar
tar -xvf VisA_20220922.tar -C datasets/VisA_20220922

# 2. Preprocess (reorganize folder structure)
python tools/prepare_visa.py \
    --data-folder datasets/VisA_20220922 \
    --save-folder datasets/VisA_pytorch
```

## Folder Structure

```text
pca-dino/
├── datasets/                   # Dataset root
│   ├── mvtec-ad/
│   └── VisA_pytorch/
├── logs/                       # Experiment logs
├── scripts/                    # SLURM/Bash benchmark scripts
│   ├── benchmark_few_shot.sh
│   ├── benchmark_full_shot.sh
│   └── ...
├── src/
│   └── subspacead/
│       ├── core/               # PCA + feature extraction code
│       ├── data/               # Dataset loaders & transforms
│       ├── post_process/       # Scoring, pixel maps, filters
│       └── utils/              # Visualization, logging
├── tools/
│   └── prepare_visa.py
├── main.py
└── README.md
```

## Usage

### Benchmark Scripts

Scripts are provided in `scripts/` to reproduce all results.
Edit `MVTEC_PATH` and `VISA_PATH` inside the scripts as needed.

**Few-Shot (1, 2, 4 Shots)**
Selects *k* normal images per category.

```bash
bash scripts/benchmark_few_shot.sh
```

**Batched Zero-Shot**
Fits PCA on the full unlabeled test set.

```bash
bash scripts/benchmark_batched0shot.sh
```

**Full-Shot**
Uses all training images.

```bash
bash scripts/benchmark_full_shot.sh
```

### Manual Execution

Example: 1-shot MVTec-AD on `bottle`:

```bash
python main.py \
    --dataset_name mvtec_ad \
    --dataset_path datasets/mvtec-ad \
    --categories bottle \
    --model_ckpt facebook/dinov2-with-registers-giant \
    --image_res 672 \
    --k_shot 1 \
    --aug_count 30 \
    --pca_ev 0.99 \
    --outdir results/debug_run
```

## Benchmark Results

### Few-Shot (Image AUROC)

| Setting    | Method         | MVTec-AD | VisA     |
| ---------- | -------------- | -------- | -------- |
| **1-Shot** | PromptAD       | 94.2     | 86.9     |
|            | AnomalyDINO    | 96.6     | 87.4     |
|            | **SubspaceAD** | **98.0** | **93.3** |
| **2-Shot** | PromptAD       | 95.7     | 88.3     |
|            | AnomalyDINO    | 96.9     | 89.7     |
|            | **SubspaceAD** | **98.1** | **94.1** |
| **4-Shot** | PromptAD       | 96.6     | 89.1     |
|            | AnomalyDINO    | 97.7     | 92.6     |
|            | **SubspaceAD** | **98.4** | **94.5** |

### Batched Zero-Shot

| Method         | MVTec-AD | VisA     |
| -------------- | -------- | -------- |
| MuSc           | **97.8** | 94.1     |
| AnomalyDINO    | 94.2     | 90.7     |
| **SubspaceAD** | 96.6     | **97.7** |

## Citation

If you find this repository useful, please consider citing:

```bibtex
@misc{lendering2026subspaceadtrainingfreefewshotanomaly,
      title={SubspaceAD: Training-Free Few-Shot Anomaly Detection via Subspace Modeling}, 
      author={Camile Lendering and Erkut Akdag and Egor Bondarev},
      year={2026},
      eprint={2602.23013},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.23013}, 
}
```
