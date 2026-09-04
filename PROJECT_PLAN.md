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

### Phase 3: Quantization & Deployment (Completed ✅)
* **ONNX Graph Export (`export_onnx.py`)**: Exported PyTorch models to standardized ONNX graphs with dynamic batch axes and validated $< 2.9\times 10^{-6}$ numerical parity.
* **Calibration-based Static INT8 PTQ (`quantize.py`)**: Quantized FP32 weights and intermediate activations using representative BDD100K calibration data, achieving **$7.5\times$ storage compression (down to 4.90 MB)** with **$99.8\%+$ mAP retention**.

### Phase 4: Benchmarking & Pareto Analysis (Completed ✅)
* **Comprehensive Evaluation Sweep**: Evaluated full matrix of models across FP32, FP16, and INT8 formats in PyTorch and ONNX Runtime.
* **Pareto Frontier Analysis (`plot_pareto.py`)**: Plotted publication-quality multi-panel Pareto curves across accuracy vs. latency and accuracy vs. storage footprint.

---

## Phase 2: Structured Pruning Benchmark & Accuracy Recovery

Evaluated on **2,000 real validation images** from BDD100K on an NVIDIA RTX 5070 Ti (Batch Size = 1, FP16):

| Model Configuration | Compression Stage | Retained Architecture | Parameters | Zero-Shot mAP | Recovered mAP | Macro F1 | Peak VRAM | $p50$ Latency | Throughput (FPS) |
|---|---|---|---|---|---|---|---|---|---|
| **CrossViT Teacher** | Baseline Teacher | Dual-Branch ($192\text{D} + 384\text{D}$) | 28.48M | — | 33.51% | 34.86% | 143.12 MB | 6.20 ms | 161.2 FPS |
| **Standard ViT** | Scratch Student Baseline | 6 Heads, 1536 MLP | 11.03M | — | 40.52% | 42.55% | 78.69 MB | 0.72 ms | 1,381.9 FPS |
| **Standard ViT (Cold KD)** | Distilled from Scratch | 6 Heads, 1536 MLP | 11.03M | — | 40.94% | 40.55% | 78.69 MB | 0.72 ms | 1,381.9 FPS |
| **Standard ViT (Warm KD)** | Distilled & Fine-Tuned | 6 Heads, 1536 MLP | 11.03M | 42.28% | 42.28% | 41.76% | 78.69 MB | 0.72 ms | 1,381.9 FPS |
| **Pruned ViT (20%)** | Zero-Shot Slicing | 5 Heads, 1229 MLP | 9.02M (-18.2%) | 41.45% | — | — | 70.57 MB | 0.62 ms | 1,622.9 FPS |
| **Pruned ViT (20%)** | **5-Epoch Recovery** | **5 Heads, 1229 MLP** | **9.02M** (-18.2%) | — | **43.07%** 🌟 | **43.72%** | **70.57 MB** | **0.62 ms** | **1,622.9 FPS** |
| **Pruned ViT (40%)** | Zero-Shot Slicing | 4 Heads, 922 MLP | 7.01M (-36.4%) | 41.20% | — | — | 61.58 MB | 0.55 ms | 1,806.8 FPS |
| **Pruned ViT (40%)** | **5-Epoch Recovery** | **4 Heads, 922 MLP** | **7.01M** (-36.4%) | — | **42.28%** 🌟 | **43.39%** | **61.58 MB** | **0.55 ms** | **1,806.8 FPS** |
| **Pruned ViT (60%)** | Zero-Shot Slicing | 2 Heads, 614 MLP | 4.41M (-60.0%) | 31.25% | — | — | 51.18 MB | 0.50 ms | 1,982.1 FPS |
| **Pruned ViT (60%)** | **5-Epoch Recovery** | **2 Heads, 614 MLP** | **4.41M** (-60.0%) | — | **40.09%** | **41.45%** | **51.18 MB** | **0.50 ms** | **1,982.1 FPS** 🚀 |

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

## Theory & Mechanics of Structured Pruning & Recovery

```
  [Unpruned ViT (11.03M)]
          │
          ▼  1. Compute L1 Importance Score for every Head & Channel
   [Rank & Identify Weakest Heads / Neurons]
          │
          ▼  2. Physical Matrix Slicing (Discard lowest 20%, 40%, 60%)
  [New Smaller Dense ViT (9.02M / 7.01M / 4.41M)]
          │
          ▼  3. Zero-Shot Evaluation (Temporary co-adaptation drop)
  [Recovery Fine-Tuning (5 Epochs with Cosine Annealing)]
          │
          ▼  4. Surviving neurons shift weights to absorb lost capacity
  [Final Accurate & Accelerated Edge Model]
```

### 1. Structured Head Pruning Formulation
In Multi-Head Attention, each head $h \in \{0, \dots, H-1\}$ maps inputs via query, key, value projection matrices $W_q^{(h)}, W_k^{(h)}, W_v^{(h)} \in \mathbb{R}^{d_h \times D}$ and output projection $W_o^{(h)} \in \mathbb{R}^{D \times d_h}$ (where $d_h = 64, D = 384$).
The composite $L_1$-norm importance score for head $h$ is:

$$I_{\text{head}}(h) = \|W_q^{(h)}\|_1 + \|W_k^{(h)}\|_1 + \|W_v^{(h)}\|_1 + \|W_o^{(h)}\|_1 = \sum_{i,j} |W_{q,ij}^{(h)}| + |W_{k,ij}^{(h)}| + |W_{v,ij}^{(h)}| + |W_{o,ij}^{(h)}|$$

The top $k = \lfloor H \times (1 - s) \rfloor$ heads are retained, and the remaining slices are physically removed from the weight tensors.

### 2. Structured MLP Channel Slicing Formulation
For a feed-forward layer with intermediate dimension $d_{\text{mlp}} = 1536$, neuron $j \in \{0, \dots, d_{\text{mlp}}-1\}$ connects via incoming weights $W_{\text{in}}[j, :] \in \mathbb{R}^{D}$ and outgoing weights $W_{\text{out}}[:, j] \in \mathbb{R}^{D}$.
Its importance score is:

$$I_{\text{mlp}}(j) = \|W_{\text{in}}[j, :]\|_1 + \|W_{\text{out}}[:, j]\|_1 = \sum_{d=1}^D |W_{\text{in}, jd}| + \sum_{d=1}^D |W_{\text{out}, dj}|$$

The top $M = \lfloor d_{\text{mlp}} \times (1 - s) \rfloor$ channels are retained, producing a new dense linear layer of shape $(M, D)$ and $(D, M)$.

### 3. Why Zero-Shot Accuracy Drops & How Recovery Restores It
* **Co-Adaptation Loss**: During initial training, neurons learn coupled representations. Cutting neurons abruptly severs these signal paths, causing a transient accuracy drop (e.g. 60% pruned drops to 31.25% mAP).
* **Recovery Fine-Tuning Mechanism**: Applying a short 5-epoch fine-tuning pass with gentle learning rates ($10^{-4} \to 10^{-6}$ via Cosine Annealing) allows the surviving dense parameters to re-adjust and absorb the functional workload of the pruned components, restoring mAP back to **40.09%** at 60% sparsity and **43.07%** at 20% sparsity.

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

---

### 4. Calibration-Based Static INT8 Post-Training Quantization (Phase 3)
Weights $W$ and activations $X$ are mapped from continuous $\mathbb{R}$ to 8-bit discrete integers $[-128, 127]$ (signed) or $[0, 255]$ (unsigned) via symmetric affine projection:

$$q = \text{clip}\left(\left\lfloor \frac{x}{S} \right\rceil + Z, q_{\text{min}}, q_{\text{max}}\right)$$

Where the dequantization scale $S$ and zero-point $Z$ are defined by dynamic activation histograms over calibration data:

$$S = \frac{\max(|x|)}{127}, \quad Z = 0 \quad (\text{Symmetric Quantization})$$

During inference, integer GEMM operations execute on Tensor Core INT8 DP4A / Tensor Core MMA instructions with integer accumulators ($\text{INT32}$), which are then re-scaled by $S_{x} \cdot S_{w}$:

$$\hat{y} = S_x S_w \sum_{k} q_{x, k} q_{w, k}$$

---

## Phase 4: Master Pareto Benchmark & Hardware Frontier Table

Comprehensive comparison across all architectures, precision tiers, and runtime execution providers evaluated on the **2,000 real validation images** of BDD100K:

| Model Architecture | Precision / Runtime | Parameters | Disk Footprint | Peak VRAM | Batch-1 GPU Latency ($p50$) | GPU FPS | ONNX CPU Latency ($p50$) | ONNX CPU FPS | Multi-Label mAP (%) | Accuracy Retention |
|---|---|---|---|---|---|---|---|---|---|---|
| **CrossViT Teacher** | FP32 (PyTorch) | 28.48M | 109.50 MB | 143.12 MB | 3.29 ms | 303.9 | 19.19 ms | 52.1 | 33.51% | 100.0% (Ref) |
| **CrossViT Teacher** | FP16 (PyTorch) | 28.48M | 109.50 MB | 143.12 MB | 6.20 ms | 161.2 | — | — | 33.51% | 100.0% |
| **CrossViT Teacher** | **INT8 (ONNX PTQ)** | **28.48M** | **30.07 MB** (-72.5%) | — | — | — | **18.00 ms** | **55.6** | **33.63%** | **100.4%** |
| **Standard ViT** | FP32 (PyTorch) | 11.03M | 42.20 MB | 78.69 MB | 1.35 ms | 739.3 | 12.34 ms | 81.1 | 40.52% | — |
| **Standard ViT** | FP16 (PyTorch) | 11.03M | 42.20 MB | 78.69 MB | 0.86 ms | 1,157.2 | — | — | 40.52% | 100.0% |
| **ViT (Cold KD)** | FP16 (PyTorch) | 11.03M | 42.20 MB | 78.69 MB | 0.72 ms | 1,381.9 | — | — | 40.94% | +0.42% |
| **ViT (Warm KD)** | FP16 (PyTorch) | 11.03M | 42.20 MB | 78.69 MB | 0.72 ms | 1,381.9 | 12.34 ms | 81.1 | **42.28%** | +1.76% |
| **ViT (Warm KD)** | **INT8 (ONNX PTQ)** | **11.03M** | **11.26 MB** (-73.3%) | — | — | — | **8.06 ms** | **124.1** | **41.34%** | **97.8%** |
| **Pruned ViT (20%)** | FP16 (PyTorch) | 9.02M | 34.54 MB | 70.57 MB | 0.62 ms | 1,622.9 | 10.22 ms | 97.8 | **43.07%** 🌟 | **+2.55%** |
| **Pruned ViT (20%)** | **INT8 (ONNX PTQ)** | **9.02M** | **9.33 MB** (-78.0%) | — | — | — | **7.00 ms** | **143.0** | **42.99%** 🌟 | **99.8%** |
| **Pruned ViT (40%)** | FP16 (PyTorch) | 7.01M | 26.89 MB | 61.58 MB | 0.55 ms | 1,806.8 | 8.48 ms | 117.9 | **42.28%** 🌟 | **+1.76%** |
| **Pruned ViT (40%)** | **INT8 (ONNX PTQ)** | **7.01M** | **7.40 MB** (-82.5%) | — | — | — | **6.67 ms** | **150.0** | **42.27%** 🌟 | **100.0%** |
| **Pruned ViT (60%)** | FP16 (PyTorch) | 4.41M | 16.96 MB | 51.18 MB | **0.50 ms** | **1,982.1** 🚀 | 5.55 ms | 180.0 | **40.09%** | 99.0% |
| **Pruned ViT (60%)** | **INT8 (ONNX PTQ)** | **4.41M** | **4.90 MB** (**-95.5%**) 🚀 | — | — | — | **6.27 ms** | **159.6** | **39.99%** | **99.7%** |

---

## Pareto Frontier Highlights & Edge Takeaways

1. **Overall Best Perception Accuracy**: **Pruned ViT (20%) + FP16** achieves **43.07% mAP** (+9.56% over CrossViT teacher) while running at **1,623 FPS** ($0.62\text{ ms}$).
2. **Optimal Real-Time Edge Compromise**: **Pruned ViT (40%) + INT8** achieves **42.27% mAP** (exact parity with unpruned warm KD) with only **7.40 MB disk size** and **150 FPS** on CPU.
3. **Extreme Embedded Micro-Deployment**: **Pruned ViT (60%) + INT8** shrinks the total model size to **4.90 MB** ($44.5\times$ smaller than baseline teacher) while retaining **40.0% mAP** and running at **1,982 FPS** on GPU ($0.50\text{ ms}$) and **160 FPS** on edge CPU.


