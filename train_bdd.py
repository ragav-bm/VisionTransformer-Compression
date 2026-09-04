import argparse
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from models import CrossViT, ViT
from dataset_bdd import BDD100KMultiLabelDataset, NUM_ATTRIBUTES, ALL_ATTRIBUTES, get_bdd_transforms
from metrics import compute_multilabel_metrics, compute_per_class_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Label Scene Attribute Perception Training (BDD100K)")
    parser.add_argument("--model", type=str, default="cvit", choices=["cvit", "vit"], help="Model architecture (cvit or vit)")
    parser.add_argument("--image-size", type=int, default=224, help="Input image resolution")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="AdamW weight decay")
    parser.add_argument("--no-amp", action="store_true", default=False, help="Disable Automatic Mixed Precision (AMP)")
    parser.add_argument("--data-dir", type=str, default="./data/bdd100k/images", help="Path to BDD100K images folder")
    parser.add_argument("--ann-file", type=str, default="./data/bdd100k/samples.json", help="Path to BDD100K JSON annotations")
    parser.add_argument("--save-dir", type=str, default="experiments", help="Directory to save checkpoints")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Quick dry run pass for debugging")
    return parser.parse_args()


def build_model(args, device):
    """Instantiates CrossViT or ViT configured for 224x224 multi-label perception (15 classes)."""
    if args.model == "cvit":
        model = CrossViT(
            image_size=args.image_size,
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
            dropout=0.1,
            emb_dropout=0.1
        )
    elif args.model == "vit":
        model = ViT(
            image_size=args.image_size,
            patch_size=16,
            num_classes=NUM_ATTRIBUTES,
            dim=384,
            depth=6,
            heads=6,
            mlp_dim=1536,
            dropout=0.1,
            emb_dropout=0.1
        )
    return model.to(device)


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, use_amp, log_interval=25):
    model.train()
    running_loss = 0.0
    total_samples = 0

    for batch_idx, (images, targets) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == len(dataloader):
            current_loss = running_loss / total_samples
            print(f"  Step [{batch_idx+1:3d}/{len(dataloader):3d}] | Batch Loss: {loss.item():.4f} | Avg Loss: {current_loss:.4f}")

    return running_loss / total_samples


def evaluate(model, dataloader, criterion, device, use_amp):
    model.eval()
    running_loss = 0.0
    total_samples = 0

    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)

            probs = torch.sigmoid(logits)

            running_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            all_targets.append(targets.cpu())
            all_probs.append(probs.cpu())

    eval_loss = running_loss / total_samples
    all_targets = torch.cat(all_targets, dim=0)
    all_probs = torch.cat(all_probs, dim=0)

    metrics_summary = compute_multilabel_metrics(all_targets, all_probs)
    metrics_summary["eval_loss"] = eval_loss
    return metrics_summary, all_targets, all_probs


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and (not args.no_amp)

    print(f"=== Starting Multi-Label Perception Training (BDD100K) ===")
    print(f"Device: {device} | AMP: {use_amp} | Architecture: {args.model.upper()} | Resolution: {args.image_size}x{args.image_size}")
    print(f"Dataset images: {args.data_dir} | Annotations: {args.ann_file}")

    # 1. Full Dataset & Train/Val Split (80% Train, 20% Validation)
    full_dataset = BDD100KMultiLabelDataset(
        image_dir=args.data_dir,
        annotation_file=args.ann_file,
        is_train=True
    )
    total_len = len(full_dataset)
    train_len = int(0.8 * total_len)
    val_len = total_len - train_len

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len], generator=generator)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    print(f"Loaded {total_len} samples: {train_len} Training | {val_len} Validation")

    # 2. Positive Weights for Class Imbalance
    pos_weights = full_dataset.calculate_pos_weights().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    # 3. Model, Optimizer, Scheduler, Scaler
    model = build_model(args, device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {total_params / 1e6:.2f}M")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    best_map = 0.0
    save_path = os.path.join(args.save_dir, f"bdd_{args.model}_best.pth")

    print(f"\nTraining for {args.epochs} epochs (Batch Size: {args.batch_size})...\n")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        print(f"--- Epoch [{epoch:02d}/{args.epochs:02d}] (lr: {scheduler.get_last_lr()[0]:.6f}) ---")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp)
        val_metrics, _, _ = evaluate(model, val_loader, criterion, device, use_amp)
        scheduler.step()
        epoch_time = time.time() - t0

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] Summary ({epoch_time:.1f}s):\n"
            f"  Train Loss : {train_loss:.4f}\n"
            f"  Val Loss   : {val_metrics['eval_loss']:.4f}\n"
            f"  Val mAP    : {val_metrics['mAP']:.2f}%\n"
            f"  Macro F1   : {val_metrics['macro_f1']:.2f}%\n"
            f"  Micro F1   : {val_metrics['micro_f1']:.2f}%"
        )

        if val_metrics["mAP"] > best_map:
            best_map = val_metrics["mAP"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "mAP": best_map,
                "config": vars(args)
            }, save_path)
            print(f"  ⭐ Saved new BEST model checkpoint -> {save_path} (mAP: {best_map:.2f}%)\n")

        if args.dry_run:
            print("[Dry Run] Exiting after 1 epoch.")
            break

    print(f"\n=== Training Complete! Best Validation mAP: {best_map:.2f}% ===")


if __name__ == "__main__":
    main()

