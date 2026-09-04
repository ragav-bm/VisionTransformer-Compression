# Vision Transformer (ViT) & Cross-ViT Compression

A PyTorch project focused on multi-label scene attribute perception and edge model compression for Vision Transformers (ViT) and Multi-Scale Cross-Attention Vision Transformers (CrossViT).

---

## Origin & Motivation

This repository builds upon an earlier baseline project: [**ragav-bm/VisionTransformer-and-Cross-ViT**](https://github.com/ragav-bm/VisionTransformer-and-Cross-ViT), which was originally developed as part of a Master's degree lab course exploring from-scratch ViT and CrossViT implementations on CIFAR-10.

Driven by personal interest to take this beyond academic toy classification, this repository explores:
1. Scaling CrossViT to **multi-label driving scene perception** (BDD100K).
2. Applying practical **edge model compression** techniques (Knowledge Distillation, Structured Pruning, INT8 Quantization) to make vision transformers viable for real-time edge deployment.

---

## Roadmap & Compression Pipeline

1. **Phase 1: Multi-Label Baseline**: Dataset pipeline for 15 BDD100K scene attributes, baseline CrossViT teacher training pipeline, and an edge latency/memory benchmarking harness.
2. **Phase 2: Knowledge Distillation & Pruning**: Transferring representations to a compact student model, followed by structured attention head and MLP channel pruning (20%, 40%, 60% sparsity) with recovery fine-tuning.
3. **Phase 3: Quantization & Deployment**: ONNX graph export, INT8 Post-Training Quantization (PTQ), and INT8 Quantization-Aware Training (QAT).
4. **Phase 4: Benchmarking & Pareto Analysis**: Comprehensive evaluation sweep measuring accuracy vs. batch-1 latency ($p50/p95$), peak memory, and parameter counts on target hardware.

For detailed milestone breakdowns, see [**`PROJECT_PLAN.md`**](PROJECT_PLAN.md).

---

## Phase 1: Core Modules & Mathematical Foundations

### 1. Multi-Label Dataset & Class Balancing ([`dataset_bdd.py`](dataset_bdd.py))
Handles 15 co-occurring driving attributes (7 Weather, 6 Scene Context, 2 Time-of-Day) at $224 \times 224$ resolution. Because rare classes (e.g. `foggy`, `snowy`, `tunnel`) appear in very few samples, we dynamically compute a positive weight vector $w_{\text{pos}, c}$:

$$w_{\text{pos}, c} = \frac{N_{\text{total}} - N_{\text{pos}, c}}{N_{\text{pos}, c}}$$

This is integrated into Weighted Binary Cross-Entropy with Logits:

$$\mathcal{L}_{\text{BCE}} = -\frac{1}{C} \sum_{c=1}^{C} \left[ w_{\text{pos}, c} \cdot y_c \log \sigma(z_c) + (1 - y_c) \log(1 - \sigma(z_c)) \right]$$

### 2. Multi-Label Evaluation Engine ([`metrics.py`](metrics.py))
Computes threshold-independent Mean Average Precision (**mAP**) and Macro/Micro F1 scores across all 15 classes:

$$\text{mAP} = \frac{1}{C} \sum_{c=1}^{C} \text{AP}_c, \quad \text{where } \text{AP}_c = \sum_{k} (R_k - R_{k-1}) P_k$$

$$\text{Macro F1} = \frac{1}{C} \sum_{c=1}^{C} \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$$

### 3. Multi-Label Teacher Training Pipeline ([`train_bdd.py`](train_bdd.py))
Trains the high-resolution CrossViT Teacher (28.48M parameters) using:
* **AdamW Decoupled Weight Decay**:
  $$\theta_{t+1} = \theta_t - \eta_t \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)$$
* **Cosine Annealing Learning Rate Schedule**:
  $$\eta_t = \eta_{\text{min}} + \frac{1}{2} (\eta_{\text{max}} - \eta_{\text{min}}) \left(1 + \cos\left(\frac{t}{T_{\text{max}}} \pi\right)\right)$$
* **Automatic Mixed Precision (AMP)** for fast 16-bit tensor operations with dynamic gradient scaling.

### 4. Edge Benchmarking Harness ([`benchmark_edge.py`](benchmark_edge.py))
Uses microsecond-precise CUDA hardware events (`torch.cuda.Event`) after a 50-iteration GPU warmup to measure real-time latency percentiles ($p50, p95, p99$), peak memory footprint, and throughput:

$$\text{Throughput (FPS)} = \frac{\text{Batch Size}}{p50 / 1000}$$

---

## Phase 1: Baseline Edge Profiling Results (RTX 5070 Ti, Batch Size = 1)

| Model Architecture | Precision | Parameters (M) | Peak VRAM (MB) | $p50$ Latency | $p95$ Latency | $p99$ Latency | Throughput (FPS) |
|---|---|---|---|---|---|---|---|
| **Standard ViT** | FP32 | 11.03M | 79.0 MB | **1.34 ms** | 1.35 ms | 1.46 ms | **744 FPS** |
| **CrossViT Teacher** | FP32 | 28.48M | 144.2 MB | **3.76 ms** | 3.87 ms | 3.97 ms | **265 FPS** |
| **CrossViT Teacher** | FP16 | 28.48M | **90.7 MB** | 9.81 ms | 10.22 ms | 10.46 ms | 102 FPS |

> **Baseline Observation**: The dual-branch CrossViT Teacher provides expressive multi-scale feature interactions, but incurs **$2.6\times$ more parameters** and **$2.8\times$ higher latency** than a single-branch ViT. This motivates our Phase 2 compression pipeline (Knowledge Distillation and Structured Pruning).

---

## Project Structure

```
├── README.md           # Project overview, mathematical formulation, and benchmarks
├── PROJECT_PLAN.md     # Detailed phase-by-phase roadmap
├── dataset_bdd.py      # BDD100K multi-label dataset loader with dynamic positive weighting
├── metrics.py          # Multi-label evaluation engine (mAP, Macro/Micro F1, threshold sweep)
├── train_bdd.py        # Multi-label teacher training pipeline with AdamW and AMP
├── benchmark_edge.py   # Edge latency, peak VRAM, and FPS benchmarking harness
├── models.py           # From-scratch ViT and CrossViT model implementations
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

### 2. Run Edge Profiling
Benchmark all models on your local GPU (batch size 1):
```bash
uv run python benchmark_edge.py --model all --batch-size 1
```

### 3. Train Multi-Label CrossViT Teacher
```bash
uv run python train_bdd.py --model cvit --epochs 10 --batch-size 32 --lr 3e-4
```
