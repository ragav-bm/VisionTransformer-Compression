import argparse
import os
import torch
import numpy as np
import onnx
import onnxruntime as ort

from models import ViT, CrossViT
from dataset_bdd import NUM_ATTRIBUTES


def parse_args():
    parser = argparse.ArgumentParser(description="ONNX Graph Export & Numerical Parity Verification")
    parser.add_argument("--model", type=str, default="all",
                        choices=["all", "warm_distilled", "pruned_20", "pruned_40", "pruned_60", "cvit"],
                        help="Model architecture to export")
    parser.add_argument("--image-size", type=int, default=224, help="Perception input image dimension")
    parser.add_argument("--opset", type=int, default=17, help="ONNX operator set version")
    parser.add_argument("--output-dir", type=str, default="experiments/onnx", help="Directory to save .onnx graphs")
    return parser.parse_args()


def load_model_and_weights(model_name, device):
    """Instantiates the specific Vision Transformer architecture and loads weights."""
    if model_name == "warm_distilled":
        model = ViT(image_size=224, patch_size=16, num_classes=NUM_ATTRIBUTES, dim=384, depth=6, heads=6, mlp_dim=1536)
        ckpt_path = "experiments/bdd_vit_warm_distilled_best.pth"
    elif model_name == "pruned_20":
        model = ViT(image_size=224, patch_size=16, num_classes=NUM_ATTRIBUTES, dim=384, depth=6, heads=5, mlp_dim=1229)
        ckpt_path = "experiments/bdd_vit_pruned_20.pth"
    elif model_name == "pruned_40":
        model = ViT(image_size=224, patch_size=16, num_classes=NUM_ATTRIBUTES, dim=384, depth=6, heads=4, mlp_dim=922)
        ckpt_path = "experiments/bdd_vit_pruned_40.pth"
    elif model_name == "pruned_60":
        model = ViT(image_size=224, patch_size=16, num_classes=NUM_ATTRIBUTES, dim=384, depth=6, heads=2, mlp_dim=614)
        ckpt_path = "experiments/bdd_vit_pruned_60.pth"
    elif model_name == "cvit":
        model = CrossViT(image_size=224, num_classes=NUM_ATTRIBUTES, sm_dim=192, lg_dim=384,
                         sm_patch_size=16, sm_enc_depth=3, sm_enc_heads=6, sm_enc_mlp_dim=768, sm_enc_dim_head=32,
                         lg_patch_size=32, lg_enc_depth=3, lg_enc_heads=6, lg_enc_mlp_dim=1536, lg_enc_dim_head=64,
                         cross_attn_depth=2, cross_attn_heads=6, cross_attn_dim_head=64, depth=3)
        ckpt_path = "experiments/bdd_cvit_best.pth"
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    if os.path.exists(ckpt_path):
        print(f"  [Load Checkpoint] Loading weights from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
    else:
        print(f"  [Warning] Checkpoint '{ckpt_path}' not found. Using initialized weights.")

    model.to(device)
    model.eval()
    return model


def export_single_model(model_name, output_dir, image_size=224, opset=17):
    """Exports a single PyTorch model to ONNX and tests numerical parity against ONNX Runtime."""
    device = torch.device("cpu")
    model = load_model_and_weights(model_name, device)

    onnx_path = os.path.join(output_dir, f"{model_name}.onnx")
    dummy_input = torch.randn(1, 3, image_size, image_size, dtype=torch.float32, device=device)

    print(f"\n📦 Exporting '{model_name}' to ONNX (Opset {opset})...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"}
        },
        dynamo=False
    )

    file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"  ✅ Saved ONNX Graph: {onnx_path} ({file_size_mb:.2f} MB)")

    # 1. Structural Schema Check
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("  ✅ ONNX Graph Schema validation passed!")

    # 2. Numerical Parity Check (PyTorch vs. ONNX Runtime)
    print("  🔍 Verifying Numerical Parity...")
    with torch.no_grad():
        torch_out = model(dummy_input).numpy()

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_inputs = {session.get_inputs()[0].name: dummy_input.numpy()}
    ort_out = session.run(None, ort_inputs)[0]

    max_diff = float(np.max(np.abs(torch_out - ort_out)))
    print(f"  • Max Absolute Difference: {max_diff:.6e}")
    assert max_diff < 1e-4, f"Parity mismatch! Diff = {max_diff}"
    print("  ✅ Parity Verified: Output matches within < 1e-4 tolerance.")

    return {
        "model": model_name,
        "size_mb": round(file_size_mb, 2),
        "max_diff": max_diff
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    models_to_export = [
        "warm_distilled",
        "pruned_20",
        "pruned_40",
        "pruned_60",
        "cvit"
    ] if args.model == "all" else [args.model]

    print("=" * 65)
    print("🚀 PHASE 3 - STEP 1: ONNX GRAPH EXPORT & VERIFICATION")
    print(f"   Target Models : {models_to_export}")
    print(f"   Opset Version : {args.opset}")
    print(f"   Output Path   : {args.output_dir}")
    print("=" * 65)

    results = []
    for m in models_to_export:
        res = export_single_model(m, args.output_dir, args.image_size, args.opset)
        results.append(res)

    print("\n" + "=" * 65)
    print("📊 EXPORT SUMMARY TABLE")
    print("=" * 65)
    print(f"| {'Model Name':<18} | {'Disk Size':<12} | {'Max Parity Diff':<18} | {'Status':<8} |")
    print(f"|{'-'*20}|{'-'*14}|{'-'*20}|{'-'*10}|")
    for r in results:
        print(f"| {r['model']:<18} | {r['size_mb']:>7.2f} MB  | {r['max_diff']:>16.2e}  | {'PASSED ✅':<8} |")
    print(f"|{'-'*20}|{'-'*14}|{'-'*20}|{'-'*10}|")


if __name__ == "__main__":
    main()

