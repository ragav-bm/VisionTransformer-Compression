# Vision Transformer (ViT) & Cross-ViT Compression

A PyTorch project exploring Vision Transformers (ViT) and Multi-Scale Cross-Attention Vision Transformers (CrossViT), progressing from baseline image classification to multi-label scene attribute perception and edge model compression.

---

## Overview

This project consists of two progressive stages:
1. **CIFAR-10 Baseline**: Implementation and training of ViT and CrossViT models on CIFAR-10.
2. **BDD100K Multi-Label Compression**: Extending CrossViT to multi-label driving scene understanding with end-to-end edge optimization (Knowledge Distillation, Structured Pruning, and INT8 Quantization).

---

## Project Structure

```
├── README.md           # Project overview and quickstart
├── PROJECT_PLAN.md     # Step-by-step roadmap and milestones
├── .gitignore          # Ignored files and artifacts
├── main.py             # CIFAR-10 training & evaluation script
└── models.py           # PyTorch ViT and CrossViT model implementations
```

---

## Quickstart (CIFAR-10)

Train CrossViT on CIFAR-10:
```bash
python main.py --model cvit --epochs 5 --batch-size 64 --lr 0.003 --aug
```

Train standard ViT:
```bash
python main.py --model vit --epochs 5 --batch-size 64 --lr 0.003 --aug
```

---

## Compression Roadmap

For the detailed multi-label compression roadmap (Knowledge Distillation, Structured Pruning, INT8 Quantization, and Edge Benchmarking), see [**`PROJECT_PLAN.md`**](file:///mnt/newssd/workspaces/VisionTransformer-and-Cross-ViT/PROJECT_PLAN.md).

---

## License
MIT License
