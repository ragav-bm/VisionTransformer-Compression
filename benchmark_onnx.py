import argparse
import os
import time
import numpy as np
import onnxruntime as ort

def benchmark_onnx_model(onnx_path, num_warmup=50, num_runs=200, providers=None):
    """
    Benchmarks ONNX Runtime inference latency and throughput.
    """
    if providers is None:
        providers = ort.get_available_providers()

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 4

    session = ort.InferenceSession(onnx_path, sess_options=opts, providers=providers)
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape

    # Handle dynamic batch size: set batch to 1
    batch_size = 1
    c = 3
    h = input_shape[2] if isinstance(input_shape[2], int) else 224
    w = input_shape[3] if isinstance(input_shape[3], int) else 224
    
    dummy_input = np.random.randn(batch_size, c, h, w).astype(np.float32)

    # 1. Warmup runs
    for _ in range(num_warmup):
        _ = session.run(None, {input_name: dummy_input})

    # 2. Timed runs
    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = session.run(None, {input_name: dummy_input})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms

    latencies = np.array(latencies)
    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    p99 = float(np.percentile(latencies, 99))
    mean_lat = float(np.mean(latencies))
    std_lat = float(np.std(latencies))
    fps = float(batch_size / (p50 / 1000.0))
    file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)

    return {
        "file_size_mb": round(file_size_mb, 2),
        "lat_p50_ms": round(p50, 2),
        "lat_p95_ms": round(p95, 2),
        "lat_p99_ms": round(p99, 2),
        "lat_mean_ms": round(mean_lat, 2),
        "lat_std_ms": round(std_lat, 2),
        "throughput_fps": round(fps, 1),
        "providers": session.get_providers()
    }


def main():
    parser = argparse.ArgumentParser(description="ONNX Runtime Edge Benchmarking Harness")
    parser.add_argument("--onnx-dir", type=str, default="experiments/onnx", help="Directory containing ONNX models")
    parser.add_argument("--runs", type=int, default=200, help="Number of timed runs")
    parser.add_argument("--warmup", type=int, default=50, help="Number of warmup runs")
    parser.add_argument("--cpu", action="store_true", default=False, help="Force CPU provider")
    args = parser.parse_args()

    providers = ["CPUExecutionProvider"] if args.cpu else ort.get_available_providers()
    print(f"=== ONNX Runtime Latency & Edge Profiling ===")
    print(f"Providers: {providers}")
    print(f"Warmup: {args.warmup} | Runs: {args.runs} | Batch Size: 1\n")

    models = [
        ("CrossViT (FP32)", "cvit.onnx"),
        ("CrossViT (INT8)", "cvit_int8.onnx"),
        ("ViT Warm KD (FP32)", "warm_distilled.onnx"),
        ("ViT Warm KD (INT8)", "warm_distilled_int8.onnx"),
        ("Pruned 20% (FP32)", "pruned_20.onnx"),
        ("Pruned 20% (INT8)", "pruned_20_int8.onnx"),
        ("Pruned 40% (FP32)", "pruned_40.onnx"),
        ("Pruned 40% (INT8)", "pruned_40_int8.onnx"),
        ("Pruned 60% (FP32)", "pruned_60.onnx"),
        ("Pruned 60% (INT8)", "pruned_60_int8.onnx"),
    ]

    header = f"| {'Model Name':<20} | {'Format':<6} | {'Disk Size':<10} | {'p50 (ms)':<8} | {'p95 (ms)':<8} | {'p99 (ms)':<8} | {'FPS':<8} |"
    divider = f"|{'-'*22}|{'-'*8}|{'-'*12}|{'-'*10}|{'-'*10}|{'-'*10}|{'-'*10}|"
    print(header)
    print(divider)

    for display_name, fname in models:
        fpath = os.path.join(args.onnx_dir, fname)
        if not os.path.exists(fpath):
            print(f"| {display_name:<20} | Missing file: {fname}")
            continue
        fmt = "INT8" if "int8" in fname else "FP32"
        res = benchmark_onnx_model(fpath, num_warmup=args.warmup, num_runs=args.runs, providers=providers)
        print(
            f"| {display_name:<20} | {fmt:<6} | {res['file_size_mb']:>6.2f} MB  | "
            f"{res['lat_p50_ms']:>6.2f} ms | {res['lat_p95_ms']:>6.2f} ms | {res['lat_p99_ms']:>6.2f} ms | {res['throughput_fps']:>7.1f}  |"
        )
    print(divider)

if __name__ == "__main__":
    main()

