import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType,
    QuantFormat,
    CalibrationMethod
)

from dataset_bdd import BDD100KMultiLabelDataset, get_bdd_transforms
from metrics import compute_multilabel_metrics


class BDD100KCalibrationDataReader(CalibrationDataReader):
    """
    Feeds real driving scene images to profile activation ranges during static calibration.
    """
    def __init__(self, dataloader, input_name="input", max_samples=100):
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


def get_file_size_mb(filepath):
    """Calculates file size in Megabytes."""
    return os.path.getsize(filepath) / (1024 * 1024)


def evaluate_onnx(onnx_path, dataloader):
    """Evaluates multi-label Mean Average Precision (mAP) using ONNX Runtime."""
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    all_targets = []
    all_probs = []

    for images, targets in dataloader:
        ort_inputs = {input_name: images.numpy().astype(np.float32)}
        logits = session.run(None, ort_inputs)[0]
        probs = 1.0 / (1.0 + np.exp(-logits))  # Sigmoid activation

        all_targets.append(targets.numpy())
        all_probs.append(probs)

    all_targets = np.concatenate(all_targets, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)

    metrics = compute_multilabel_metrics(all_targets, all_probs, threshold=0.5)
    return metrics


def quantize_model(model_name, onnx_dir, calib_loader, num_calib=100):
    """Applies Static INT8 Post-Training Quantization (PTQ) with Calibration."""
    fp32_path = os.path.join(onnx_dir, f"{model_name}.onnx")
    int8_path = os.path.join(onnx_dir, f"{model_name}_int8.onnx")

    if not os.path.exists(fp32_path):
        raise FileNotFoundError(f"Missing FP32 ONNX model: '{fp32_path}'. Run export_onnx.py first!")

    fp32_size = get_file_size_mb(fp32_path)
    print(f"\n⚙️  Quantizing '{model_name}' (Original FP32 Size: {fp32_size:.2f} MB)...")

    # 1. Create Calibration Reader
    calib_reader = BDD100KCalibrationDataReader(calib_loader, input_name="input", max_samples=num_calib)

    # 2. Run Static PTQ
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

    int8_size = get_file_size_mb(int8_path)
    compression_ratio = fp32_size / int8_size
    print(f"  ✅ Quantization Completed: {fp32_size:.2f} MB ➔ {int8_size:.2f} MB ({compression_ratio:.1f}x compression!)")

    return fp32_path, int8_path, fp32_size, int8_size


def parse_args():
    parser = argparse.ArgumentParser(description="Calibration-Based Static INT8 Post-Training Quantization")
    parser.add_argument("--model", type=str, default="all",
                        choices=["all", "warm_distilled", "pruned_20", "pruned_40", "pruned_60", "cvit"],
                        help="Model to quantize")
    parser.add_argument("--num-calib", type=int, default=100, help="Number of calibration driving scene images")
    parser.add_argument("--onnx-dir", type=str, default="experiments/onnx", help="Directory containing .onnx files")
    parser.add_argument("--data-dir", type=str, default="./data/bdd100k/images", help="Path to BDD100K images")
    parser.add_argument("--ann-file", type=str, default="./data/bdd100k/samples.json", help="Path to BDD100K annotations")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("💎 PHASE 3 - STEP 2: STATIC INT8 POST-TRAINING QUANTIZATION (PTQ)")
    print(f"   Calibration Samples : {args.num_calib} real driving images")
    print(f"   ONNX Directory      : {args.onnx_dir}")
    print("=" * 70)

    # 1. Prepare Calibration and Validation Datasets
    print("\n[Data Pipeline] Preparing BDD100K validation and calibration sets...")
    val_dataset = BDD100KMultiLabelDataset(
        image_dir=args.data_dir,
        annotation_file=args.ann_file,
        transform=get_bdd_transforms(image_size=224, is_train=False),
        is_train=False
    )
    generator = torch.Generator().manual_seed(42)
    _, val_set = random_split(val_dataset, [8000, 2000], generator=generator)

    calib_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

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

        print(f"  🔍 Evaluating perception mAP on 2,000 validation images...")
        fp32_metrics = evaluate_onnx(fp32_path, val_loader)
        int8_metrics = evaluate_onnx(int8_path, val_loader)

        map_drop = fp32_metrics["mAP"] - int8_metrics["mAP"]
        retention = (int8_metrics["mAP"] / fp32_metrics["mAP"]) * 100

        summary_rows.append({
            "model": m,
            "fp32_size": fp32_size,
            "int8_size": int8_size,
            "ratio": fp32_size / int8_size,
            "fp32_map": fp32_metrics["mAP"],
            "int8_map": int8_metrics["mAP"],
            "retention": retention
        })

        print(f"  • FP32 mAP: {fp32_metrics['mAP']:.2f}% | INT8 mAP: {int8_metrics['mAP']:.2f}% (Retention: {retention:.1f}%)")

    print("\n" + "=" * 80)
    print("📊 INT8 QUANTIZATION SUMMARY & ACCURACY RETENTION")
    print("=" * 80)
    print(f"| {'Model Name':<16} | {'FP32 Size':<11} | {'INT8 Size':<11} | {'Ratio':<7} | {'FP32 mAP':<10} | {'INT8 mAP':<10} | {'Retention':<10} |")
    print(f"|{'-'*18}|{'-'*13}|{'-'*13}|{'-'*9}|{'-'*12}|{'-'*12}|{'-'*12}|")
    for r in summary_rows:
        print(
            f"| {r['model']:<16} | {r['fp32_size']:>8.2f} MB | {r['int8_size']:>8.2f} MB | {r['ratio']:>5.1f}x | "
            f"{r['fp32_map']:>8.2f}% | {r['int8_map']:>8.2f}% | {r['retention']:>9.1f}% |"
        )
    print(f"|{'-'*18}|{'-'*13}|{'-'*13}|{'-'*9}|{'-'*12}|{'-'*12}|{'-'*12}|")


if __name__ == "__main__":
    main()

