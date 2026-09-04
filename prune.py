import argparse
import os
import time
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from models import ViT, Attention, FeedForward, PreNorm
from dataset_bdd import BDD100KMultiLabelDataset, NUM_ATTRIBUTES, ALL_ATTRIBUTES, get_bdd_transforms
from metrics import compute_multilabel_metrics, compute_per_class_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Structured Head & Channel Pruning for Vision Transformers")
    parser.add_argument("--base-ckpt", type=str, default="experiments/bdd_vit_warm_distilled_best.pth",
                        help="Path to unpruned student checkpoint (Warm Distilled or Scratch ViT)")
    parser.add_argument("--sparsity", type=float, default=0.20, choices=[0.20, 0.40, 0.60],
                        help="Target pruning sparsity ratio (0.20, 0.40, 0.60)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of recovery fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for fine-tuning")
    parser.add_argument("--lr", type=float, default=1e-4, help="Fine-tuning learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="AdamW weight decay")
    parser.add_argument("--data-dir", type=str, default="./data/bdd100k/images", help="Path to BDD100K images folder")
    parser.add_argument("--ann-file", type=str, default="./data/bdd100k/samples.json", help="Path to BDD100K JSON annotations")
    parser.add_argument("--save-dir", type=str, default="experiments", help="Directory to save checkpoints")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Quick dry run pass for debugging")
    return parser.parse_args()


def load_base_model(ckpt_path, device):
    """Loads the unpruned baseline student ViT from checkpoint."""
    model = ViT(
        image_size=224,
        patch_size=16,
        num_classes=NUM_ATTRIBUTES,
        dim=384,
        depth=6,
        heads=6,
        mlp_dim=1536
    ).to(device)

    if os.path.exists(ckpt_path):
        print(f"[Pruning] Loading base weights from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    model.eval()
    return model


def prune_vit_structured(base_model, sparsity_ratio, device):
    """
    Physically prunes attention heads and MLP hidden dimensions based on L1-norm importance scoring.
    Returns a new, smaller dense ViT model with transferred weights.
    """
    orig_heads = 6
    orig_mlp_dim = 1536
    dim = 384
    dim_head = 64
    depth = 6

    # Calculate target dimensions
    pruned_heads = max(1, round(orig_heads * (1.0 - sparsity_ratio)))
    pruned_mlp_dim = max(64, round(orig_mlp_dim * (1.0 - sparsity_ratio)))

    print("\n" + "="*70)
    print(f"✂️  STRUCTURED PRUNING (Target Sparsity: {int(sparsity_ratio * 100)}%)")
    print(f"   • Attention Heads  : {orig_heads} heads -> {pruned_heads} heads (pruning {orig_heads - pruned_heads} heads/layer)")
    print(f"   • MLP Intermediate : {orig_mlp_dim} -> {pruned_mlp_dim} channels (pruning {orig_mlp_dim - pruned_mlp_dim} channels/layer)")
    print("="*70)

    # Instantiate the new smaller dense ViT architecture
    pruned_model = ViT(
        image_size=224,
        patch_size=16,
        num_classes=NUM_ATTRIBUTES,
        dim=dim,
        depth=depth,
        heads=pruned_heads,
        mlp_dim=pruned_mlp_dim
    ).to(device)

    # 1. Copy patch embedding, pos embedding, cls token, and classification head directly
    pruned_model.to_patch_embedding.load_state_dict(base_model.to_patch_embedding.state_dict())
    pruned_model.pos_embedding.data.copy_(base_model.pos_embedding.data)
    pruned_model.cls_token.data.copy_(base_model.cls_token.data)
    pruned_model.mlp_head.load_state_dict(base_model.mlp_head.state_dict())

    # 2. Layer-by-layer structured weight slicing
    for layer_idx in range(depth):
        orig_prenorm_attn, orig_prenorm_ff = base_model.transformer.layers[layer_idx]
        new_prenorm_attn, new_prenorm_ff = pruned_model.transformer.layers[layer_idx]

        # Copy PreNorm LayerNorm weights
        new_prenorm_attn.norm.load_state_dict(orig_prenorm_attn.norm.state_dict())
        new_prenorm_ff.norm.load_state_dict(orig_prenorm_ff.norm.state_dict())

        # --- A. ATTENTION HEAD PRUNING ---
        orig_attn = orig_prenorm_attn.fn
        new_attn = new_prenorm_attn.fn

        # Compute L1-norm importance for each head
        head_scores = []
        for h in range(orig_heads):
            start = h * dim_head
            end = (h + 1) * dim_head
            q_norm = orig_attn.q1.weight[start:end, :].abs().sum().item()
            k_norm = orig_attn.k1.weight[start:end, :].abs().sum().item()
            v_norm = orig_attn.v1.weight[start:end, :].abs().sum().item()
            o_norm = orig_attn.out1[0].weight[:, start:end].abs().sum().item()
            head_scores.append((q_norm + k_norm + v_norm + o_norm, h))

        # Rank heads by importance (highest first) and keep top `pruned_heads`
        head_scores.sort(key=lambda x: x[0], reverse=True)
        selected_heads = sorted([h for _, h in head_scores[:pruned_heads]])
        pruned_out_heads = [h for _, h in head_scores[pruned_heads:]]
        print(f"  Layer {layer_idx + 1}/6: Retained Heads {selected_heads}, Pruned Heads {pruned_out_heads}")

        # Build channel indices for selected heads
        selected_attn_channels = []
        for h in selected_heads:
            selected_attn_channels.extend(range(h * dim_head, (h + 1) * dim_head))
        selected_attn_channels = torch.tensor(selected_attn_channels, dtype=torch.long, device=device)

        # Slice and copy Q, K, V projections and Output projection
        new_attn.q1.weight.data.copy_(orig_attn.q1.weight.data[selected_attn_channels, :])
        if orig_attn.q1.bias is not None:
            new_attn.q1.bias.data.copy_(orig_attn.q1.bias.data[selected_attn_channels])

        new_attn.k1.weight.data.copy_(orig_attn.k1.weight.data[selected_attn_channels, :])
        if orig_attn.k1.bias is not None:
            new_attn.k1.bias.data.copy_(orig_attn.k1.bias.data[selected_attn_channels])

        new_attn.v1.weight.data.copy_(orig_attn.v1.weight.data[selected_attn_channels, :])
        if orig_attn.v1.bias is not None:
            new_attn.v1.bias.data.copy_(orig_attn.v1.bias.data[selected_attn_channels])

        new_attn.out1[0].weight.data.copy_(orig_attn.out1[0].weight.data[:, selected_attn_channels])
        if orig_attn.out1[0].bias is not None:
            new_attn.out1[0].bias.data.copy_(orig_attn.out1[0].bias.data)

        # --- B. MLP CHANNEL PRUNING ---
        orig_ff = orig_prenorm_ff.fn
        new_ff = new_prenorm_ff.fn

        w_in = orig_ff.net[0].weight.data    # (1536, 384)
        w_out = orig_ff.net[3].weight.data   # (384, 1536)

        # Compute L1 importance for each intermediate channel
        channel_importance = w_in.abs().sum(dim=1) + w_out.abs().sum(dim=0)
        _, top_channels = torch.topk(channel_importance, k=pruned_mlp_dim, largest=True)
        top_channels, _ = torch.sort(top_channels)

        # Slice and copy MLP weights
        new_ff.net[0].weight.data.copy_(w_in[top_channels, :])
        if orig_ff.net[0].bias is not None:
            new_ff.net[0].bias.data.copy_(orig_ff.net[0].bias.data[top_channels])

        new_ff.net[3].weight.data.copy_(w_out[:, top_channels])
        if orig_ff.net[3].bias is not None:
            new_ff.net[3].bias.data.copy_(orig_ff.net[3].bias.data)

    # Calculate parameter count comparison
    base_params = sum(p.numel() for p in base_model.parameters()) / 1e6
    pruned_params = sum(p.numel() for p in pruned_model.parameters()) / 1e6
    reduction = ((base_params - pruned_params) / base_params) * 100.0

    print(f"\n✅ Slicing Complete: {base_params:.2f}M -> {pruned_params:.2f}M params ({reduction:.1f}% reduction)")
    return pruned_model


def evaluate_model(model, val_loader, device):
    """Evaluates multi-label perception metrics on validation set."""
    model.eval()
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.sigmoid(logits)
            all_targets.append(targets.cpu())
            all_probs.append(probs.cpu())

    all_targets = torch.cat(all_targets, dim=0)
    all_probs = torch.cat(all_probs, dim=0)
    metrics = compute_multilabel_metrics(all_targets, all_probs, threshold=0.5)
    return metrics, all_targets, all_probs


def fine_tune_pruned(pruned_model, train_loader, val_loader, pos_weights, args, device):
    """Executes recovery fine-tuning with cosine annealing and weighted BCE."""
    print(f"\n🔄 Starting Recovery Fine-Tuning ({args.epochs} Epochs, lr={args.lr:.1e})...")

    pos_weights = pos_weights.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer = AdamW(pruned_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')

    best_mAP = 0.0
    best_state = None
    pct = int(args.sparsity * 100)
    save_path = os.path.join(args.save_dir, f"bdd_vit_pruned_{pct}.pth")

    for epoch in range(1, args.epochs + 1):
        pruned_model.train()
        total_loss = 0.0
        start_time = time.time()

        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits = pruned_model(images)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            if args.dry_run and batch_idx >= 5:
                break

        scheduler.step()
        train_time = time.time() - start_time
        avg_loss = total_loss / (len(train_loader) if not args.dry_run else 6)

        # Validation evaluation
        val_metrics, _, _ = evaluate_model(pruned_model, val_loader, device)
        val_mAP = val_metrics["mAP"]
        val_recall = val_metrics["macro_recall"]
        current_lr = scheduler.get_last_lr()[0]

        is_best = val_mAP > best_mAP
        if is_best:
            best_mAP = val_mAP
            best_state = copy.deepcopy(pruned_model.state_dict())
            torch.save({
                "epoch": epoch,
                "model_state_dict": best_state,
                "mAP": best_mAP,
                "sparsity": args.sparsity,
                "heads": pruned_model.transformer.layers[0][0].fn.heads,
                "mlp_dim": pruned_model.transformer.layers[0][1].fn.net[0].out_features,
            }, save_path)

        star = " 🌟 [BEST]" if is_best else ""
        print(f"  Epoch [{epoch:02d}/{args.epochs:02d}] ({train_time:.1f}s) | "
              f"Train Loss: {avg_loss:.4f} | "
              f"Val mAP: {val_mAP:.2f}% | "
              f"Val Recall: {val_recall:.2f}% | "
              f"LR: {current_lr:.2e}{star}")

        if args.dry_run:
            break

    print(f"\n💾 Saved best pruned model ({pct}% sparsity) to: {save_path} (Best mAP: {best_mAP:.2f}%)")
    return best_mAP, save_path


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Structured Pruning Engine (Target Sparsity: {int(args.sparsity * 100)}%) ===")
    print(f"Hardware Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    os.makedirs(args.save_dir, exist_ok=True)

    # 1. Load Dataset
    print("\n[Data Pipeline] Loading BDD100K dataset...")
    full_dataset = BDD100KMultiLabelDataset(
        image_dir=args.data_dir,
        annotation_file=args.ann_file,
        transform=get_bdd_transforms(image_size=224, is_train=True),
        is_train=True
    )
    val_dataset = BDD100KMultiLabelDataset(
        image_dir=args.data_dir,
        annotation_file=args.ann_file,
        transform=get_bdd_transforms(image_size=224, is_train=False),
        is_train=False
    )

    generator = torch.Generator().manual_seed(42)
    train_set, _ = random_split(full_dataset, [8000, 2000], generator=generator)
    _, val_set = random_split(val_dataset, [8000, 2000], generator=generator)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    pos_weights = full_dataset.calculate_pos_weights()

    # 2. Load Base Model and evaluate baseline
    base_model = load_base_model(args.base_ckpt, device)
    print("\n[Baseline Evaluation] Evaluating unpruned model...")
    base_metrics, _, _ = evaluate_model(base_model, val_loader, device)
    print(f"  Unpruned Base Model -> mAP: {base_metrics['mAP']:.2f}%, Recall: {base_metrics['macro_recall']:.2f}%")

    # 3. Apply Structured Pruning
    pruned_model = prune_vit_structured(base_model, args.sparsity, device)

    # 4. Evaluate Zero-Shot Pruned Accuracy (before recovery fine-tuning)
    print("\n[Zero-Shot Evaluation] Evaluating immediately after weight slicing (no fine-tuning)...")
    zero_shot_metrics, _, _ = evaluate_model(pruned_model, val_loader, device)
    print(f"  Zero-Shot Pruned Model -> mAP: {zero_shot_metrics['mAP']:.2f}%, Recall: {zero_shot_metrics['macro_recall']:.2f}%")

    # 5. Recovery Fine-Tuning
    best_mAP, save_path = fine_tune_pruned(pruned_model, train_loader, val_loader, pos_weights, args, device)

    # 6. Final Evaluation & Per-Class Summary
    best_ckpt = torch.load(save_path, map_location=device, weights_only=False)
    pruned_model.load_state_dict(best_ckpt["model_state_dict"])
    final_metrics, targets, probs = evaluate_model(pruned_model, val_loader, device)
    per_class = compute_per_class_metrics(targets, probs, threshold=0.5, class_names=ALL_ATTRIBUTES)

    print("\n" + "="*70)
    print(f"🎯 PRUNING SUMMARY ({int(args.sparsity * 100)}% Sparsity)")
    print(f"   • Base mAP (Unpruned)  : {base_metrics['mAP']:.2f}%")
    print(f"   • Zero-Shot Pruned mAP : {zero_shot_metrics['mAP']:.2f}%")
    print(f"   • Recovered Best mAP   : {final_metrics['mAP']:.2f}%")
    print(f"   • Recovered Recall     : {final_metrics['macro_recall']:.2f}%")
    print("="*70)


if __name__ == "__main__":
    main()

