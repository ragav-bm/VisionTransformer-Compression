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

## Phase 2: Knowledge Distillation & Structured Pruning Results

Evaluated on **2,000 real validation images** from BDD100K across all 15 driving scene attributes on an NVIDIA RTX 5070 Ti (Batch Size = 1, FP16):

| Model Configuration | Strategy / Stage | Parameters | Zero-Shot mAP | Recovered mAP | Peak VRAM | $p50$ Latency | Throughput (FPS) |
|---|---|---|---|---|---|---|---|
| **CrossViT Teacher** | Baseline Teacher | 28.48M | — | 33.51% | 143.12 MB | 6.20 ms | 161.2 FPS |
| **Standard ViT** | Scratch Baseline Student | 11.03M | — | 40.52% | 78.69 MB | 0.72 ms | 1,381.9 FPS |
| **Standard ViT (Cold KD)** | Distilled from Scratch | 11.03M | — | 40.94% | 78.69 MB | 0.72 ms | 1,381.9 FPS |
| **Standard ViT (Warm KD)** | **Distilled & Fine-Tuned** | **11.03M** | 42.28% | **42.28%** 🚀 | 78.69 MB | 0.72 ms | 1,381.9 FPS |
| **Pruned ViT (20%)** | Zero-Shot Slicing (5H, 1229 MLP) | 9.02M (-18.2%) | 41.45% | — | 70.57 MB | 0.62 ms | 1,622.9 FPS |
| **Pruned ViT (20%)** | **5-Epoch Recovery Fine-Tuned** | **9.02M** (-18.2%) | — | **43.07%** 🌟 | **70.57 MB** | **0.62 ms** | **1,622.9 FPS** |
| **Pruned ViT (40%)** | Zero-Shot Slicing (4H, 922 MLP) | 7.01M (-36.4%) | 41.20% | — | 61.58 MB | 0.55 ms | 1,806.8 FPS |
| **Pruned ViT (40%)** | **5-Epoch Recovery Fine-Tuned** | **7.01M** (-36.4%) | — | **42.28%** 🌟 | **61.58 MB** | **0.55 ms** | **1,806.8 FPS** |
| **Pruned ViT (60%)** | Zero-Shot Slicing (2H, 614 MLP) | 4.41M (-60.0%) | 31.25% | — | 51.18 MB | 0.50 ms | 1,982.1 FPS |
| **Pruned ViT (60%)** | **5-Epoch Recovery Fine-Tuned** | **4.41M** (-60.0%) | — | **40.09%** | **51.18 MB** | **0.50 ms** | **1,982.1 FPS** 🚀 |

* **Cold vs. Warm Distillation in a single line**: Cold-start distillation trains an uninitialized student from scratch with teacher soft guidance (40.94% mAP), whereas Warm-start distillation fine-tunes an already-converged student to refine decision boundaries on tail attributes (42.28% mAP, +1.76% gain).
* **Slicing vs. Recovery in a single line**: Physical dense slicing removes redundant heads and neurons causing a transient co-adaptation drop (Zero-Shot), which a brief 5-epoch fine-tuning pass completely recovers (up to 43.07% mAP) by allowing surviving parameters to absorb the lost capacity.

> 💡 **Pruning Theory & Hardware Insights**: For the complete mathematical formulations of $L_1$-norm head/channel importance scoring, FlashAttention kernel fusion, and co-adaptation recovery, see [**`PROJECT_PLAN.md` (Theory & Mechanics of Structured Pruning)**](PROJECT_PLAN.md#theory--mechanics-of-structured-pruning--recovery).
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
