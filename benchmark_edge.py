import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn

from models import CrossViT, ViT
from dataset_bdd import NUM_ATTRIBUTES


def count_parameters(model):
    """Calculates total, trainable, and non-trainable parameters in millions."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params / 1e6, trainable_params / 1e6


def measure_peak_memory(model, input_tensor, device):
    """
    Measures the peak CUDA memory allocated during a single forward inference pass (in MB).
    """
    if device.type != "cuda":
        return 0.0

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        _ = model(input_tensor)

    peak_bytes = torch.cuda.max_memory_allocated(device)
    return peak_bytes / (1024 * 1024)  # Convert to MB


def benchmark_latency(
    model,
    input_shape=(1, 3, 224, 224),
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    num_warmup=50,
    num_runs=200,
    use_fp16=False
):
    """
    Accurately benchmarks inference latency (p50, p95, p99, FPS) using CUDA Events.
    
    Args:
        model (nn.Module): PyTorch model to benchmark
        input_shape (tuple): Shape of dummy input tensor (B, C, H, W)
        device (torch.device): CUDA or CPU device
        num_warmup (int): Number of warmup iterations to stabilize GPU clocks & cache
        num_runs (int): Number of timed iterations to build latency distribution
        use_fp16 (bool): Whether to evaluate using half-precision (FP16)
        
    Returns:
        dict: Benchmarking summary (p50, p95, p99, mean, std, fps, peak_memory_mb, params_m)
    """
    model.eval()
    model.to(device)

    if use_fp16 and device.type == "cuda":
        model = model.half()
        dummy_input = torch.randn(*input_shape, dtype=torch.float16, device=device)
    else:
        dummy_input = torch.randn(*input_shape, dtype=torch.float32, device=device)

    batch_size = input_shape[0]
    total_m, trainable_m = count_parameters(model)

    # 1. Warmup Runs (stabilizes GPU power state, cuDNN heuristics, and caches)
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    # 2. Timed Runs using CUDA Events (microsecond precision)
    latencies = []

    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        with torch.no_grad():
            for _ in range(num_runs):
                start_event.record()
                _ = model(dummy_input)
                end_event.record()
                torch.cuda.synchronize(device)

                elapsed_ms = start_event.elapsed_time(end_event)
                latencies.append(elapsed_ms)
    else:
        # Fallback for CPU benchmarking using high-resolution monotonic clock
        with torch.no_grad():
            for _ in range(num_runs):
                t0 = time.perf_counter()
                _ = model(dummy_input)
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)

    latencies = np.array(latencies)

    # 3. Compute Latency Distribution & Throughput
    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    p99 = float(np.percentile(latencies, 99))
    mean_lat = float(np.mean(latencies))
    std_lat = float(np.std(latencies))
    fps = float((batch_size / (p50 / 1000.0)))

    # 4. Measure Peak Memory
    peak_mem = measure_peak_memory(model, dummy_input, device)

    return {
        "batch_size": batch_size,
        "precision": "FP16" if use_fp16 else "FP32",
        "params_m": round(total_m, 2),
        "peak_mem_mb": round(peak_mem, 2),
        "lat_p50_ms": round(p50, 2),
        "lat_p95_ms": round(p95, 2),
        "lat_p99_ms": round(p99, 2),
        "lat_mean_ms": round(mean_lat, 2),
        "lat_std_ms": round(std_lat, 2),
        "throughput_fps": round(fps, 1)
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Edge Latency & Resource Profiling Harness")
    parser.add_argument("--model", type=str, default="cvit", choices=["cvit", "vit", "all"], help="Model to profile")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for latency measurement (1 for real-time edge)")
    parser.add_argument("--image-size", type=int, default=224, help="Input image dimension (e.g. 224)")
    parser.add_argument("--runs", type=int, default=200, help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=50, help="Number of warmup iterations")
    parser.add_argument("--fp16", action="store_true", default=False, help="Run in half-precision FP16 mode")
    return parser.parse_args()


def build_benchmark_model(model_type, image_size=224):
    """Instantiates model architecture for profiling."""
    if model_type == "cvit":
        return CrossViT(
            image_size=image_size,
            num_classes=NUM_ATTRIBUTES,
            sm_dim=192,
            lg_dim=384,
            sm_patch_size=16,
            sm_enc_depth=3,
            sm_enc_heads=6,
            sm_enc_mlp_dim=768,
            sm_enc_dim_head=32,
            lg_patch_size=32,
            lg_enc_depth=3,
            lg_enc_heads=6,
            lg_enc_mlp_dim=1536,
            lg_enc_dim_head=64,
            cross_attn_depth=2,
            cross_attn_heads=6,
            cross_attn_dim_head=64,
            depth=3,
            dropout=0.0,
            emb_dropout=0.0
        )
    elif model_type == "vit":
        return ViT(
            image_size=image_size,
            patch_size=16,
            num_classes=NUM_ATTRIBUTES,
            dim=384,
            depth=6,
            heads=6,
            mlp_dim=1536,
            dropout=0.0,
            emb_dropout=0.0
        )


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"=== Edge Profiling Harness (Device: {device.type.upper()}) ===")
    print(f"Config: Batch Size={args.batch_size} | Resolution={args.image_size}x{args.image_size} | Warmup={args.warmup} | Runs={args.runs}\n")

    models_to_test = ["cvit", "vit"] if args.model == "all" else [args.model]

    header = f"| {'Model':<12} | {'Precision':<9} | {'Params (M)':<10} | {'Peak RAM (MB)':<13} | {'p50 (ms)':<8} | {'p95 (ms)':<8} | {'p99 (ms)':<8} | {'FPS':<7} |"
    divider = f"|{'-'*14}|{'-'*11}|{'-'*12}|{'-'*15}|{'-'*10}|{'-'*10}|{'-'*10}|{'-'*9}|"
    print(header)
    print(divider)

    for m_type in models_to_test:
        model = build_benchmark_model(m_type, image_size=args.image_size)
        res = benchmark_latency(
            model,
            input_shape=(args.batch_size, 3, args.image_size, args.image_size),
            device=device,
            num_warmup=args.warmup,
            num_runs=args.runs,
            use_fp16=args.fp16
        )

        name = "CrossViT" if m_type == "cvit" else "Standard ViT"
        print(
            f"| {name:<12} | {res['precision']:<9} | {res['params_m']:<10} | {res['peak_mem_mb']:<13} | "
            f"{res['lat_p50_ms']:<8} | {res['lat_p95_ms']:<8} | {res['lat_p99_ms']:<8} | {res['throughput_fps']:<7} |"
        )
    print(divider)
