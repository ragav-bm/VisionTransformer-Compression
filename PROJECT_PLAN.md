# Project Plan: Cross-ViT Compression & Edge Optimization

A personal research and engineering project exploring Vision Transformers (ViT) and Cross-Attention Vision Transformers (CrossViT), scaling from baseline image classification to multi-label scene attribute perception and model compression for edge deployment.

---

## Roadmap

### Phase 1: Data & Baseline 
- **Multi-Label Dataset**: BDD100K scene attributes (15 binary labels: weather, scene context, time of day) with class-imbalance weighting (`pos_weight`).
- **Teacher Baseline**: Train and evaluate a baseline CrossViT model using Mean Average Precision (mAP).
- **Edge Benchmark Harness**: Measure batch-1 latency ($p50/p95$), peak memory, and parameter counts.

### Phase 2: Knowledge Distillation & Pruning
- **Knowledge Distillation**: Transfer dark knowledge from the CrossViT teacher to a compact DeiT student using temperature-scaled soft loss.
- **Structured Pruning**: Prune redundant attention heads and MLP channels at 20%, 40%, and 60% sparsity levels.
- **Recovery Fine-Tuning**: Fine-tune pruned models to recover lost perception performance.

### Phase 3: Quantization & Deployment
- **ONNX Export**: Export models to standard ONNX format.
- **INT8 PTQ**: Post-Training Quantization with calibration.
- **INT8 QAT**: Quantization-Aware Training with fake-quantization layers and fine-tuning.

### Phase 4: Benchmarking & Evaluation
- **Evaluation Sweep**: Benchmark all model configurations (Teacher, Student, Distilled, Pruned, Quantized).
- **Pareto Analysis**: Analyze accuracy vs. latency trade-offs on edge devices.
