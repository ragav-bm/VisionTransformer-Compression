# Vision Transformer (ViT) & Cross-ViT Compression

A PyTorch project focused on multi-label scene attribute perception and edge model compression for Vision Transformers (ViT) and Multi-Scale Cross-Attention Vision Transformers (CrossViT).

---

## Origin & Motivation

This repository builds upon an earlier baseline project: [**ragav-bm/VisionTransformer-and-Cross-ViT**](https://github.com/ragav-bm/VisionTransformer-and-Cross-ViT), which was originally developed as part of a Master's degree lab course exploring from-scratch ViT and CrossViT implementations on CIFAR-10.

Driven by personal interest to take this beyond academic toy classification, this repository explores:
1. Scaling CrossViT to **multi-label driving scene perception** (BDD100K).
2. Applying practical **edge model compression** techniques (Knowledge Distillation, Structured Pruning, INT8 Quantization) to make vision transformers viable for real-time edge deployment.

---

## Phase 1 Hardware Profiling & Latency Breakdown (Batch Size = 1)

Evaluated at full perception resolution ($224 \times 224$) on an NVIDIA RTX 5070 Ti under real-time batch-1 streaming conditions using hardware CUDA events over 200 iterations (after 50 warmup iterations):

| Model Architecture | Precision | Parameters | Peak VRAM | $p50$ Latency | $p95$ Latency | $p99$ Latency | Throughput (FPS) |
|---|---|---|---|---|---|---|---|
| **Standard ViT** | **FP32** | 11.03M | 78.69 MB | 1.35 ms | 1.39 ms | 1.59 ms | **739.3 FPS** |
| **Standard ViT** | **FP16** | 11.03M | 78.69 MB | **0.86 ms** | **0.92 ms** | **0.99 ms** | **1,157.2 FPS** 🚀 |
| **CrossViT Teacher** | **FP32** | 28.48M | 143.12 MB | 3.29 ms | 3.84 ms | 4.48 ms | **303.9 FPS** |
| **CrossViT Teacher** | **FP16** | 28.48M | 143.12 MB | 5.54 ms | 6.00 ms | 6.09 ms | **180.4 FPS** |

---

## Phase 2: Knowledge Distillation Results

Evaluated on **2,000 real validation images** from BDD100K across all 15 driving scene attributes:

| Model Configuration | Training Strategy | Parameters | mAP (%) | Macro Recall (%) | $p50$ Latency (FP16) | Throughput (FPS) |
|---|---|---|---|---|---|---|
| **CrossViT Teacher** | Multi-Scale Baseline | 28.48M | 33.51% | 56.94% | 5.54 ms | 180.4 FPS |
| **Standard ViT** | Scratch Baseline Student | 11.03M | 40.52% | 62.92% | **0.86 ms** | **1,157.2 FPS** |
| **Standard ViT (Cold KD)** | Distilled from Scratch | 11.03M | 40.94% | **65.24%** | **0.86 ms** | **1,157.2 FPS** |
| **Standard ViT (Warm KD)** | **Fine-Tuned with Teacher KD** | **11.03M** | **42.28%** 🚀 | 64.65% | **0.86 ms** | **1,157.2 FPS** |

* **Cold vs. Warm Distillation in a single line**: Cold-start distillation trains an uninitialized student from scratch with teacher soft guidance (40.94% mAP), whereas Warm-start distillation fine-tunes an already-converged student to refine decision boundaries on tail attributes (42.28% mAP, +1.76% gain).

> 💡 **Hardware Insights & Optimizations**: For details on FlashAttention-2 integration and why the single-stream ViT achieves 1,157 FPS while dual-stream CrossViT is CPU-dispatch bound at batch 1, see [**`PROJECT_PLAN.md` (Code Optimizations & Hardware Insights)**](PROJECT_PLAN.md#code-optimizations--hardware-insights).
>
> 📊 **Detailed Per-Class Breakdown**: For the full 15-attribute per-class AP breakdown (e.g. `parking lot` +9.69%, `rainy` +4.96%, `residential` +3.07%), see [**`PROJECT_PLAN.md` (Per-Class AP Report)**](PROJECT_PLAN.md#phase-2-comprehensive-per-class-average-precision-ap--report).
---

## Roadmap & Compression Pipeline

1. **Phase 1: Multi-Label Baseline** (Completed ✅): Dataset pipeline for 15 BDD100K scene attributes, CrossViT teacher baseline training pipeline, and edge profiling harness.
2. **Phase 2: Knowledge Distillation & Structured Pruning** (Completed ✅): Multi-scale KD (+1.76% mAP) and structured head/channel pruning (20%, 40%, 60% sparsity) achieving up to **1,982 FPS** (0.50 ms) at **4.41M parameters**.
3. **Phase 3: Quantization & Deployment** (Next 🎯): ONNX graph export, INT8 Post-Training Quantization (PTQ), and INT8 Quantization-Aware Training (QAT).
4. **Phase 4: Benchmarking & Pareto Analysis**: Comprehensive evaluation sweep measuring mAP accuracy vs. batch-1 latency ($p50$) to identify optimal deployment candidates for edge devices.

For detailed milestone breakdowns, see [**`PROJECT_PLAN.md`**](PROJECT_PLAN.md).

---

## Project Structure

```
├── README.md           # Project overview, benchmark results, and engineering insights
├── PROJECT_PLAN.md     # Full 4-phase roadmap and mathematical formulations
├── dataset_bdd.py      # BDD100K multi-label dataset loader with dynamic positive weighting
├── metrics.py          # Multi-label evaluation engine (mAP, Macro/Micro F1, threshold sweep)
├── train_bdd.py        # Multi-label teacher & baseline student training pipeline
├── distill.py          # Knowledge distillation pipeline (supports cold and warm start)
├── prune.py            # Structured head and channel pruning with recovery fine-tuning
├── benchmark_edge.py   # Edge latency, peak VRAM, and FPS benchmarking harness
├── models.py           # Hardware-optimized ViT and CrossViT model implementations
├── main.py             # Legacy CIFAR-10 classification script
├── assets/             # Benchmark plots and visual assets
└── experiments/        # Saved model weights and experiment checkpoints (gitignored)
```

---

## Quickstart

### 1. Requirements Setup
```bash
uv add torch torchvision einops scikit-learn pandas matplotlib
```

### 2. Edge Latency Benchmarking
Profile batch-1 inference across all architectures:
```bash
# FP32 Benchmark
uv run python benchmark_edge.py --model all --batch-size 1

# Hardware FP16 Benchmark
uv run python benchmark_edge.py --model all --batch-size 1 --fp16
```

### 3. Multi-Label Training
Train the CrossViT Teacher or Standard ViT Student:
```bash
# Train Teacher (CrossViT)
uv run python train_bdd.py --model cvit --epochs 10 --batch-size 32 --lr 3e-4

# Train Student Baseline (Standard ViT)
uv run python train_bdd.py --model vit --epochs 10 --batch-size 32 --lr 3e-4
```

### 4. Warm-Start Knowledge Distillation
Distill the CrossViT teacher into the pre-trained ViT student:
```bash
uv run python distill.py --teacher-ckpt experiments/bdd_cvit_best.pth --student-ckpt experiments/bdd_vit_best.pth --epochs 10 --batch-size 32 --lr 1e-4 --temperature 3.0 --alpha 0.5
```

### 5. Structured Head & Channel Pruning
Prune attention heads and MLP channels across sparsity tiers:
```bash
# 20% Sparsity (9.02M params, 1,623 FPS)
uv run python prune.py --sparsity 0.20 --epochs 5

# 40% Sparsity (7.01M params, 1,807 FPS)
uv run python prune.py --sparsity 0.40 --epochs 5

# 60% Sparsity (4.41M params, 1,982 FPS)
uv run python prune.py --sparsity 0.60 --epochs 5
```
