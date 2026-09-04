import argparse
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_static,
    quantize_dynamic,
    CalibrationDataReader,
    QuantType,
    QuantFormat,
    CalibrationMethod
)

from dataset_bdd import BDD100KMultiLabelDataset, NUM_ATTRIBUTES, ALL_ATTRIBUTES, get_bdd_transforms
from metrics import compute_multilabel_metrics, compute_per_class_metrics


class BDD100KCalibrationDataReader(CalibrationDataReader):
    """Feeds real driving scene images to calibrate activation quantization scales."""
    def __init__(self, dataloader, input_name="input", max_samples=200):
        self.dataloader = dataloader
        self.input_name = input_name
        self.max_samples = max_samples
        self.data_iter = iter(self.dataloader)
        self.samples_yielded = 0

    def get_next(self):
        if self.samples_yielded >= self.max_samples:
            return None
        try:
            images, _ = next(self.data_iter)
            self.samples_yielded += images.size(0)
            return {self.input_name: images.numpy().astype(np.float32)}
        except StopIteration:
            return None

    def rewind(self):
        self.data_iter = iter(self.dataloader)
        self.samples_yielded = 0


def get_model_size_mb(onnx_path):
    """Computes total on-disk size including external data files."""
    total_bytes = os.path.getsize(onnx_path)
    data_path = f"{onnx_path}.data"
    if os.path.exists(data_path):
        total_bytes += os.path.getsize(data_path)
    return total_bytes / (1024 * 1024)


def evaluate_onnx_model(onnx_path, dataloader):
    """Evaluates multi-label perception metrics using ONNX Runtime."""
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    all_targets = []
    all_probs = []

    for images, targets in dataloader:
        ort_inputs = {input_name: images.numpy().astype(np.float32)}
        logits = session.run(None, ort_inputs)[0]
        probs = 1.0 / (1.0 + np.exp(-logits))  # Sigmoid in numpy

        all_targets.append(targets.numpy())
        all_probs.append(probs)

    all_targets = np.concatenate(all_targets, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)

    metrics = compute_multilabel_metrics(all_targets, all_probs, threshold=0.5)
    return metrics, all_targets, all_probs


def quantize_model(model_name, onnx_dir, calib_loader, num_calib=200):
    """Applies INT8 Post-Training Quantization (PTQ) with activation calibration."""
    fp32_path = os.path.join(onnx_dir, f"{model_name}.onnx")
    int8_path = os.path.join(onnx_dir, f"{model_name}_int8.onnx")

    if not os.path.exists(fp32_path):
        raise FileNotFoundError(f"FP32 ONNX model not found: {fp32_path}. Run export_onnx.py first.")

    fp32_size = get_model_size_mb(fp32_path)
    print(f"\n⚙️  Quantizing '{model_name}' (FP32 Size: {fp32_size:.2f} MB)...")

    # Static Calibration Data Reader
    calib_reader = BDD100KCalibrationDataReader(calib_loader, input_name="input", max_samples=num_calib)

    try:
        # Static PTQ (Quantizes both weights and activations)
        quantize_static(
            model_input=fp32_path,
            model_output=int8_path,
            calibration_data_reader=calib_reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            calibrate_method=CalibrationMethod.MinMax
        )
    except Exception as e:
        print(f"  [Warning] Static quantization hit an issue ({e}). Falling back to dynamic PTQ...")
        quantize_dynamic(
            model_input=fp32_path,
            model_output=int8_path,
            weight_type=QuantType.QInt8,
            per_channel=True
        )

    int8_size = get_model_size_mb(int8_path)
    compression_ratio = fp32_size / max(int8_size, 1e-4)
    print(f"  ✅ Quantization Complete: {fp32_size:.2f} MB -> {int8_size:.2f} MB ({compression_ratio:.1f}x compression!)")

    return fp32_path, int8_path, fp32_size, int8_size


def parse_args():
    parser = argparse.ArgumentParser(description="INT8 Post-Training Quantization (PTQ) with Calibration")
    parser.add_argument("--model", type=str, default="all",
                        choices=["all", "warm_distilled", "pruned_20", "pruned_40", "pruned_60", "cvit"],
                        help="Model to quantize")
    parser.add_argument("--num-calib", type=int, default=200, help="Number of calibration images for static PTQ")
    parser.add_argument("--onnx-dir", type=str, default="experiments/onnx", help="Directory containing ONNX models")
    parser.add_argument("--data-dir", type=str, default="./data/bdd100k/images", help="Path to BDD100K images folder")
    parser.add_argument("--ann-file", type=str, default="./data/bdd100k/samples.json", help="Path to BDD100K JSON annotations")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 75)
    print("💎 INT8 POST-TRAINING QUANTIZATION (PTQ) PIPELINE")
    print(f"   Calibration Samples : {args.num_calib} driving images")
    print(f"   ONNX Directory      : {args.onnx_dir}")
    print("=" * 75)

    # 1. Prepare Validation & Calibration Loaders
    print("\n[Data Pipeline] Loading BDD100K validation dataset...")
    val_dataset = BDD100KMultiLabelDataset(
        image_dir=args.data_dir,
        annotation_file=args.ann_file,
        transform=get_bdd_transforms(image_size=224, is_train=False),
        is_train=False
    )
    generator = torch.Generator().manual_seed(42)
    _, val_set = random_split(val_dataset, [8000, 2000], generator=generator)

    calib_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=4)

    models_to_quantize = [
        "warm_distilled",
        "pruned_20",
        "pruned_40",
        "pruned_60",
        "cvit"
    ] if args.model == "all" else [args.model]

    summary_rows = []

    for m in models_to_quantize:
        fp32_path, int8_path, fp32_size, int8_size = quantize_model(
            m, args.onnx_dir, calib_loader, num_calib=args.num_calib
        )

        print(f"  🔍 Evaluating FP32 vs. INT8 accuracy on 2,000 validation images...")
        fp32_metrics, _, _ = evaluate_onnx_model(fp32_path, val_loader)
        int8_metrics, _, _ = evaluate_onnx_model(int8_path, val_loader)

        map_drop = fp32_metrics["mAP"] - int8_metrics["mAP"]

        summary_rows.append({
            "model": m,
            "fp32_size": fp32_size,
            "int8_size": int8_size,
            "compression": fp32_size / max(int8_size, 1e-4),
            "fp32_map": fp32_metrics["mAP"],
            "int8_map": int8_metrics["mAP"],
            "map_drop": map_drop
        })

        print(f"  • FP32 mAP : {fp32_metrics['mAP']:.2f}% | INT8 mAP : {int8_metrics['mAP']:.2f}% (Drop: {map_drop:+.2f}%)")

    print("\n" + "=" * 80)
    print("📊 INT8 QUANTIZATION SUMMARY & ACCURACY RETENTION")
    print("=" * 80)
    header = f"| {'Model':<16} | {'FP32 Size':<11} | {'INT8 Size':<11} | {'Ratio':<7} | {'FP32 mAP':<10} | {'INT8 mAP':<10} | {'Retention':<10} |"
    divider = f"|{'-'*18}|{'-'*13}|{'-'*13}|{'-'*9}|{'-'*12}|{'-'*12}|{'-'*12}|"
    print(header)
    print(divider)
    for r in summary_rows:
        retention = f"{r['int8_map'] / max(r['fp32_map'], 1e-4) * 100:.1f}%"
        print(
            f"| {r['model']:<16} | {r['fp32_size']:>8.2f} MB | {r['int8_size']:>8.2f} MB | {r['compression']:>5.1f}x | "
            f"{r['fp32_map']:>8.2f}% | {r['int8_map']:>8.2f}% | {retention:>10} |"
        )
    print(divider)


if __name__ == "__main__":
    main()
