import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from models import CrossViT, ViT
from dataset_bdd import BDD100KMultiLabelDataset, NUM_ATTRIBUTES, ALL_ATTRIBUTES
from metrics import compute_multilabel_metrics, compute_per_class_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Label Knowledge Distillation (CrossViT -> ViT Student)")
    parser.add_argument("--teacher-ckpt", type=str, default="experiments/bdd_cvit_best.pth", help="Path to trained Teacher checkpoint")
    parser.add_argument("--student-ckpt", type=str, default=None, help="Optional path to pre-trained Student weights for warm-start distillation fine-tuning")
    parser.add_argument("--image-size", type=int, default=224, help="Input image resolution")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of distillation epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Student initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="AdamW weight decay")
    parser.add_argument("--temperature", type=float, default=3.0, help="Distillation temperature T (softens logits)")
    parser.add_argument("--alpha", type=float, default=0.5, help="Balancing weight: (1-alpha)*HardLoss + alpha*SoftLoss")
    parser.add_argument("--no-amp", action="store_true", default=False, help="Disable Automatic Mixed Precision")
    parser.add_argument("--data-dir", type=str, default="./data/bdd100k/images", help="Path to BDD100K images folder")
    parser.add_argument("--ann-file", type=str, default="./data/bdd100k/samples.json", help="Path to BDD100K JSON annotations")
    parser.add_argument("--save-dir", type=str, default="experiments", help="Directory to save checkpoints")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Quick dry run pass for debugging")
    return parser.parse_args()


class MultiLabelDistillationLoss(nn.Module):
    """
    Composite Multi-Label Distillation Loss:
    L_total = (1 - alpha) * L_hard + alpha * (T^2) * L_soft
    """
    def __init__(self, pos_weights, temperature=3.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.hard_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    def forward(self, student_logits, teacher_logits, targets):
        # 1. Hard Ground-Truth Loss (with positive weights for class imbalance)
        loss_hard = self.hard_criterion(student_logits, targets)

        # 2. Soft Knowledge Distillation Loss
        T = self.temperature
        student_soft = student_logits / T
        teacher_soft = torch.sigmoid(teacher_logits / T)

        # Multi-label soft binary cross-entropy
        loss_soft = F.binary_cross_entropy_with_logits(student_soft, teacher_soft)

        # Scale soft loss by T^2 as per standard distillation theory (Hinton et al.)
        loss_total = (1.0 - self.alpha) * loss_hard + (self.alpha * (T ** 2)) * loss_soft
        return loss_total, loss_hard, loss_soft


def load_teacher_model(ckpt_path, device, image_size=224):
    """Loads the pre-trained CrossViT Teacher model and freezes all weights."""
    teacher = CrossViT(
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
        depth=3
    )

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        teacher.load_state_dict(state_dict)
        print(f"✅ Loaded Teacher weights from '{ckpt_path}'")
    else:
        print(f"⚠️ Teacher checkpoint '{ckpt_path}' not found! Using initialized weights.")

    teacher.to(device).eval()
    for param in teacher.parameters():
        param.requires_grad = False

    return teacher


def build_student_model(device, image_size=224, init_ckpt=None):
    """Instantiates the compact ViT Student, optionally loading pre-trained weights."""
    student = ViT(
        image_size=image_size,
        patch_size=16,
        num_classes=NUM_ATTRIBUTES,
        dim=384,
        depth=6,
        heads=6,
        mlp_dim=1536,
        dropout=0.1,
        emb_dropout=0.1
    )

    if init_ckpt and os.path.exists(init_ckpt):
        ckpt = torch.load(init_ckpt, map_location=device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        student.load_state_dict(state_dict)
        print(f"🔥 Warm-Start: Loaded pre-trained Student weights from '{init_ckpt}'")
    else:
        print(f"❄️ Cold-Start: Training Student from scratch with Teacher guidance.")

    return student.to(device)


def train_distill_epoch(student, teacher, dataloader, criterion, optimizer, scaler, device, use_amp, log_interval=25):
    student.train()
    running_loss = 0.0
    running_hard = 0.0
    running_soft = 0.0
    total_samples = 0

    for batch_idx, (images, targets) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=use_amp):
                teacher_logits = teacher(images)

        with torch.amp.autocast('cuda', enabled=use_amp):
            student_logits = student(images)
            total_loss, loss_hard, loss_soft = criterion(student_logits, teacher_logits, targets)

        if use_amp:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()

        running_loss += total_loss.item() * images.size(0)
        running_hard += loss_hard.item() * images.size(0)
        running_soft += loss_soft.item() * images.size(0)
        total_samples += images.size(0)

        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == len(dataloader):
            avg_loss = running_loss / total_samples
            avg_hard = running_hard / total_samples
            avg_soft = running_soft / total_samples
            print(f"  Step [{batch_idx+1:3d}/{len(dataloader):3d}] | Total Loss: {avg_loss:.4f} (Hard: {avg_hard:.4f}, Soft: {avg_soft:.4f})")

    return running_loss / total_samples


def evaluate_student(student, dataloader, criterion, device, use_amp):
    student.eval()
    running_loss = 0.0
    total_samples = 0
    all_targets, all_probs = [], []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = student(images)
                loss = criterion.hard_criterion(logits, targets)

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
    return metrics_summary


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and (not args.no_amp)

    print(f"=== Starting Multi-Label Knowledge Distillation Pipeline ===")
    print(f"Device: {device} | AMP: {use_amp} | Temperature (T): {args.temperature} | Alpha: {args.alpha}")

    # 1. Dataset & DataLoader (80/20 split)
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

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    print(f"Loaded {total_len} samples: {train_len} Training | {val_len} Validation")

    # 2. Positive Weights & Distillation Criterion
    pos_weights = full_dataset.calculate_pos_weights().to(device)
    distill_criterion = MultiLabelDistillationLoss(pos_weights=pos_weights, temperature=args.temperature, alpha=args.alpha)

    # 3. Models Setup
    teacher = load_teacher_model(args.teacher_ckpt, device, image_size=args.image_size)
    student = build_student_model(device, image_size=args.image_size, init_ckpt=args.student_ckpt)

    teacher_params = sum(p.numel() for p in teacher.parameters()) / 1e6
    student_params = sum(p.numel() for p in student.parameters() if p.requires_grad) / 1e6
    print(f"Teacher Parameters: {teacher_params:.2f}M (Frozen 🔒)")
    print(f"Student Parameters: {student_params:.2f}M (Trainable 🎯 - 2.6x smaller!)")

    # 4. Optimizer, Scheduler, Scaler
    optimizer = AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    best_map = 0.0
    if args.save_name:
    print(f"\nDistilling for {args.epochs} epochs (Batch Size: {args.batch_size})...\n")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        print(f"--- Distillation Epoch [{epoch:02d}/{args.epochs:02d}] (lr: {scheduler.get_last_lr()[0]:.6f}) ---")
        train_loss = train_distill_epoch(student, teacher, train_loader, distill_criterion, optimizer, scaler, device, use_amp)
        val_metrics = evaluate_student(student, val_loader, distill_criterion, device, use_amp)
        scheduler.step()
        epoch_time = time.time() - t0

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] Summary ({epoch_time:.1f}s):\n"
            f"  Train Total Loss : {train_loss:.4f}\n"
            f"  Val Loss         : {val_metrics['eval_loss']:.4f}\n"
            f"  Val mAP          : {val_metrics['mAP']:.2f}%\n"
            f"  Macro F1         : {val_metrics['macro_f1']:.2f}%\n"
            f"  Micro F1         : {val_metrics['micro_f1']:.2f}%"
        )

        if val_metrics["mAP"] > best_map:
            best_map = val_metrics["mAP"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "mAP": best_map,
                "temperature": args.temperature,
                "alpha": args.alpha,
                "config": vars(args)
            }, save_path)
            print(f"  ⭐ Saved new BEST Distilled Student model -> {save_path} (mAP: {best_map:.2f}%)\n")

        if args.dry_run:
            print("[Dry Run] Exiting after 1 epoch.")
            break

    print(f"\n=== Knowledge Distillation Complete! Best Validation mAP: {best_map:.2f}% ===")


if __name__ == "__main__":
    main()

