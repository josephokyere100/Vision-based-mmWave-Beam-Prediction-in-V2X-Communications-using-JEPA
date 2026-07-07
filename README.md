# Vision-based mmWave Beam Prediction in V2X Communications using JEPA

A JEPA-based self-supervised foundational model pretrained on unlabeled vehicular camera images for mmWave beam prediction in V2X systems. A single frozen encoder serves multiple downstream tasks including beam prediction, LOS/NLOS detection, and day/night classification without retraining.

## Requirements

- Python 3.10
- PyTorch >= 2.0
- torchvision
- scikit-learn
- pandas
- numpy
- matplotlib
- Pillow
- pyyaml

## Dataset

**DeepSense 6G** — Download from https://deepsense6g.net

**e-FLASH** — Download from https://www.wi-lab.net/e-flash

The dataset is not included in this repository due to its size. After downloading, update the file paths in the notebook accordingly.

## Pretraining

The ViT-Base encoder is pretrained using I-JEPA on approximately 112,000 unlabeled camera images from the DeepSense 6G dataset across 23 driving scenarios. Pretraining was run on the ARC HPC cluster using an NVIDIA A100 GPU.

```bash
python main.py --fname configs/deepsense6G_config.yaml --devices cuda:0
```

## Running the Code

All downstream experiments are in the Colab notebook:

**`ENEL_619_07_Final_Code_Joseph.ipynb`**

Open it in Google Colab and run each section in order:

1. Setup and mount Google Drive
2. Pretraining — skip this if the checkpoint already exists
3. Beam Prediction — frozen I-JEPA encoder
4. Beam Prediction — ResNet18 supervised baseline
5. Day/Night Classification — frozen I-JEPA encoder
6. LOS/NLOS Classification — ResNet18
7. Few-Shot Beam Prediction — I-JEPA vs ResNet18 on e-FLASH

## Pretraining Configuration

| Parameter | Value |
|---|---|
| Architecture | ViT-Base |
| Patch size | 16 × 16 pixels |
| Embedding dimension | 768 |
| Transformer blocks | 12 |
| Pretraining epochs | 100 |
| Batch size | 256 |
| Optimizer | AdamW |
| Hardware | NVIDIA A100 — ARC HPC Cluster |

## Acknowledgements

- I-JEPA: Assran et al., 2023 — https://arxiv.org/abs/2301.08243
- DeepSense 6G Dataset — https://deepsense6g.net
- e-FLASH Dataset — University of Southern California Wireless Intelligence Lab
