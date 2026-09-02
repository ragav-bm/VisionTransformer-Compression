# Vision Transformer (ViT) & Cross-ViT Compression

A PyTorch project focused on multi-label scene attribute perception and edge model compression for Vision Transformers (ViT) and Multi-Scale Cross-Attention Vision Transformers (CrossViT).

---

## Origin & Motivation

This repository builds upon an earlier baseline project: [**ragav-bm/VisionTransformer-and-Cross-ViT**](https://github.com/ragav-bm/VisionTransformer-and-Cross-ViT), which was originally developed as part of a Master's degree lab course exploring from-scratch ViT and CrossViT implementations on CIFAR-10.

Driven by personal interest to take this beyond academic toy classification, this repository explores:
1. Scaling CrossViT to **multi-label driving scene perception** (BDD100K).
2. Applying practical **edge model compression** techniques to make vision transformers viable for low-latency and resource-constrained edge deployment.

---

## Roadmap & Compression Pipeline

This project is organized into four sequential phases:

1. **Phase 1: Multi-Label Baseline**: Dataset pipeline for BDD100K scene attributes (weather, scene, time-of-day) with positive weighting, baseline CrossViT teacher training, and an edge latency benchmarking harness.
2. **Phase 2: Knowledge Distillation & Pruning**: Transferring dark knowledge to a compact DeiT student model, followed by structured attention head and MLP channel pruning (20%, 40%, 60%) with recovery fine-tuning.
3. **Phase 3: Quantization & Deployment**: ONNX graph export, INT8 Post-Training Quantization (PTQ), and INT8 Quantization-Aware Training (QAT).
4. **Phase 4: Benchmarking & Pareto Analysis**: Comprehensive evaluation sweep measuring accuracy vs. batch-1 latency ($p50/p95$), peak memory, and parameter counts on target edge hardware.

For detailed milestone breakdowns, see [**`PROJECT_PLAN.md`**](PROJECT_PLAN.md).

---

## Project Structure

```
├── README.md           # Project overview and scope
├── PROJECT_PLAN.md     # Detailed phase-by-phase roadmap
├── main.py             # Baseline training and evaluation script
├── models.py           # ViT and CrossViT model architectures
├── assets/             # Benchmark plots and visual assets
└── experiments/        # Saved model weights and experiment logs (gitignored)
```

---

## Quickstart

### 1. Requirements Setup (via `uv`)
```bash
uv add torch torchvision einops matplotlib
```
*Or using standard pip:*
```bash
pip install torch torchvision einops matplotlib
```

### 2. Baseline Training Commands (CIFAR-10)
Train CrossViT baseline:
```bash
uv run python main.py --model cvit --epochs 5 --batch-size 64 --lr 0.003 --aug
```

Train standard ViT:
```bash
uv run python main.py --model vit --epochs 5 --batch-size 64 --lr 0.003 --aug
```

Train ResNet-18 baseline:
```bash
uv run python main.py --model r18 --epochs 5 --batch-size 64 --lr 0.003 --aug
```
