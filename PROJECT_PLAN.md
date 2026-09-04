# Project Plan: Cross-ViT Compression & Edge Optimization

A research and engineering roadmap exploring Vision Transformers (ViT) and Multi-Scale Cross-Attention Vision Transformers (CrossViT), scaling from baseline classification to multi-label scene attribute perception and edge model compression.

---

## Roadmap & Milestone Breakdown

### Phase 1: Data & Baseline (Completed ✅)
* **15-Attribute Dataset Pipeline (`dataset_bdd.py`)**: Multi-label driving scene perception across Weather (7), Scene (6), and Time-of-Day (2) with positive weight balancing on 10,000 real images.
* **Evaluation Engine (`metrics.py`)**: Threshold-independent Mean Average Precision (mAP), Macro/Micro F1, and per-class performance tracking.
* **Teacher Training Pipeline (`train_bdd.py`)**: High-resolution $224 \times 224$ CrossViT Teacher (28.48M params) with AdamW, Cosine Annealing, and AMP.
* **Edge Profiling Harness (`benchmark_edge.py`)**: Microsecond-precise CUDA hardware timing for batch-1 latency ($p50, p95, p99$), peak VRAM, and FPS.

### Phase 2: Knowledge Distillation & Structured Pruning (Completed ✅)
* **Multi-Label Knowledge Distillation (`distill.py`) (Completed ✅)**: Transferred multi-scale representations from the CrossViT Teacher into the fast ViT Student using composite weighted BCE and temperature-scaled distillation loss ($T=3.0, \alpha=0.5$), boosting mAP to **42.28%**.
* **Structured Head & Channel Pruning (`prune.py`) (Completed ✅)**: Sliced redundant Multi-Head Attention heads and intermediate MLP hidden dimensions across 20%, 40%, and 60% sparsity levels using L1-norm importance criteria, achieving up to a **$1.44\times$ hardware speedup (0.50 ms / 1,982 FPS)** and **60.0% parameter reduction** (11.03M $\to$ 4.41M).
* **Recovery Fine-Tuning (Completed ✅)**: Restored perception accuracy across all sparsity tiers using 5-epoch cosine-annealed fine-tuning on BDD100K.

### Phase 3: Quantization & Deployment (Next 🎯)
* **ONNX Graph Export (`export_onnx.py`)**: Export PyTorch model graphs to standardized ONNX representation with fixed batch-1 inference shapes.
* **INT8 Post-Training Quantization (PTQ)**: Quantize FP32 weights and activations to INT8 using calibration data to determine optimal quantization scales.
* **INT8 Quantization-Aware Training (QAT)**: Insert fake-quantization operators during fine-tuning to model 8-bit rounding errors and preserve mAP.

### Phase 4: Benchmarking & Pareto Analysis
* **Comprehensive Evaluation Sweep**: Benchmark all model configurations (Teacher, Student, Distilled, Pruned at 20/40/60%, PTQ INT8, QAT INT8).
* **Pareto Frontier Analysis**: Plot mAP accuracy vs. batch-1 latency ($p50$) to identify optimal deployment candidates for edge devices.

---

## Phase 2: Structured Pruning Benchmark & Accuracy Recovery

Evaluated on **2,000 real validation images** from BDD100K on an NVIDIA RTX 5070 Ti (Batch Size = 1, FP16):

| Model Configuration | Sparsity | Parameters | Zero-Shot mAP | Recovered mAP | Peak VRAM | $p50$ Latency | Throughput (FPS) |
|---|---|---|---|---|---|---|---|
| **Unpruned Warm ViT** | 0% | 11.03M | 54.96% | 54.96% | 78.69 MB | 0.72 ms | 1,381.9 FPS |
| **Pruned ViT (20%)** | 20% | **9.02M** (-18.2%) | 53.90% | **60.16%** 🌟 | **70.57 MB** | **0.62 ms** | **1,622.9 FPS** |
| **Pruned ViT (40%)** | 40% | **7.01M** (-36.4%) | 53.69% | **60.40%** 🌟 | **61.58 MB** | **0.55 ms** | **1,806.8 FPS** |
| **Pruned ViT (60%)** | 60% | **4.41M** (-60.0%) | 40.70% | **54.74%** | **51.18 MB** | **0.50 ms** | **1,982.1 FPS** 🚀 |

---

## Phase 2: Comprehensive Per-Class Average Precision (AP %) Report

Evaluated on the **2,000 real validation images** from BDD100K:

| Attribute | Category | Teacher (28.5M) AP | Scratch ViT (11.0M) AP | Warm Distilled ViT AP | Improvement with KD |
|---|---|---|---|---|---|
| **`parking lot`** | Scene | 2.45% | 4.39% | **14.08%** | **+9.69%** 🌟 (3.2x gain!) |
| **`rainy`** | Weather | 14.98% | 26.08% | **31.04%** | **+4.96%** 🟢 |
| **`residential`** | Scene | 16.42% | 19.21% | **22.28%** | **+3.07%** 🟢 |
| **`highway`** | Scene | 37.87% | 52.71% | **55.28%** | **+2.57%** 🟢 |
| **`overcast`** | Weather | 29.47% | 40.23% | **42.74%** | **+2.51%** 🟢 |
| **`snowy`** | Weather | 7.24% | 14.68% | **17.10%** | **+2.42%** 🟢 |
| **`undefined`** | Weather | 26.91% | 49.12% | **50.93%** | **+1.81%** 🟢 |
| **`city street`** | Scene | 72.37% | 75.72% | **76.88%** | **+1.16%** 🟢 |
| **`partly cloudy`** | Weather | 16.80% | 21.06% | **21.74%** | **+0.68%** 🟢 |
| **`tunnel`** | Scene | 15.08% | 27.47% | **27.66%** | **+0.19%** 🟢 |
| **`daytime`** | TimeOfDay | 90.27% | 92.40% | **92.39%** | -0.01% |
| **`foggy`** | Weather | 0.50% | 1.19% | **0.85%** | -0.34% |
| **`night`** | TimeOfDay | 95.98% | 98.66% | **97.70%** | -0.96% |
| **`clear`** | Weather | 76.16% | 84.76% | **83.33%** | -1.43% |
| **`gas stations`** | Scene | 0.00% | 0.00% | **0.00%** | +0.00% |
| **OVERALL mAP** | — | **33.50%** | **40.51%** | **42.27%** | **+1.76% Overall Gain** 🚀 |

---

## Code Optimizations & Hardware Insights

### 1. FlashAttention-2 Kernel Fusion
In [`models.py`](models.py), attention operations were upgraded from naive `torch.einsum` matrix multiplications to PyTorch's native `F.scaled_dot_product_attention` (FlashAttention-2). This optimization:
* Directs computation to dedicated NVIDIA Tensor Cores (FP16 / BF16).
* Fuses $QK^T$ scaling, softmax, and dropout into a single GPU SRAM kernel pass, eliminating intermediate $O(N^2)$ memory writes.

### 2. Architectural Bottleneck Analysis: ViT vs. CrossViT on Tensor Cores
* **Standard ViT**: Operates on a single contiguous token sequence $(1, 197, 384)$. Tensor Cores execute uninterrupted without host synchronization barriers, yielding a **$1.57\times$ FP16 acceleration (0.86 ms / 1,157 FPS)**.
* **CrossViT Teacher**: Possesses asymmetric dual branches ($196 \text{ tokens} \neq 49 \text{ tokens}$, $192\text{D} \neq 384\text{D}$). Each attention layer requires token slicing (`[:, :1]`), dimension projection adapters (`ProjectInOut`), and host-triggered concatenations (`torch.cat`). At batch size 1, CPU kernel dispatch overhead exceeds raw GPU compute time, starving Tensor Cores and leading to $5.54\text{ ms}$ latency.
* **Distillation Rationale**: Knowledge Distillation allows transferring the representational capability of the multi-scale teacher into the streamlined single-sequence ViT student, retaining high perception accuracy while executing at real-time edge speeds ($0.86\text{ ms}$).

---

## Mathematical Foundations & Formulations

### 1. Dynamic Positive Class Weighting
$$w_{\text{pos}, c} = \frac{N_{\text{total}} - N_{\text{pos}, c}}{N_{\text{pos}, c}}$$

$$\mathcal{L}_{\text{BCE}} = -\frac{1}{C} \sum_{c=1}^{C} \left[ w_{\text{pos}, c} \cdot y_c \log \sigma(z_c) + (1 - y_c) \log(1 - \sigma(z_c)) \right]$$

---

### 2. Multi-Label Knowledge Distillation Loss (Phase 2)
Combines hard ground-truth supervision with temperature-scaled teacher soft targets:

$$\mathcal{L}_{\text{total}} = (1 - \alpha) \mathcal{L}_{\text{BCE}}(z_s, y) + \alpha \cdot T^2 \cdot \mathcal{L}_{\text{BCE}}\left(\frac{z_s}{T}, \sigma\left(\frac{z_t}{T}\right)\right)$$

* $T = 3.0$ softens output logits into dark knowledge probabilities.
* $\alpha = 0.5$ balances ground-truth annotations with teacher guidance.
* $T^2 = 9.0$ ensures gradient scale invariance when dividing logits by $T$.

---

### 3. Mean Average Precision (mAP) & Real-Time Hardware Timing
$$\text{AP}_c = \sum_{k} (R_k - R_{k-1}) P_k, \quad \text{mAP} = \frac{1}{C} \sum_{c=1}^{C} \text{AP}_c$$

$$\text{Throughput (FPS)} = \frac{\text{Batch Size}}{p50 / 1000}$$

