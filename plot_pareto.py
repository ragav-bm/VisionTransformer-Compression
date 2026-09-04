import os
import matplotlib.pyplot as plt
import numpy as np

def generate_pareto_plots(output_dir="assets"):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "pareto_frontier.png")

    # Set style
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

    # Data definition
    # (Name, mAP, Latency_ms_GPU, Disk_Size_MB, Params_M, Type, Color, Marker)
    data = [
        {"name": "CrossViT Teacher (FP32)", "map": 33.51, "lat_gpu": 3.29, "disk_mb": 109.50, "params_m": 28.48, "tier": "Teacher", "color": "#7f7f7f", "marker": "s"},
        {"name": "CrossViT Teacher (INT8)", "map": 33.63, "lat_gpu": 3.20, "disk_mb": 30.07, "params_m": 28.48, "tier": "Teacher INT8", "color": "#a55194", "marker": "D"},
        {"name": "ViT Scratch Baseline",   "map": 40.52, "lat_gpu": 0.72, "disk_mb": 42.20, "params_m": 11.03, "tier": "Baseline", "color": "#3182bd", "marker": "o"},
        {"name": "ViT Cold KD",            "map": 40.94, "lat_gpu": 0.72, "disk_mb": 42.20, "params_m": 11.03, "tier": "Cold KD", "color": "#6baed6", "marker": "o"},
        {"name": "ViT Warm KD (FP16)",      "map": 42.28, "lat_gpu": 0.72, "disk_mb": 42.20, "params_m": 11.03, "tier": "Warm KD", "color": "#2ca02c", "marker": "P"},
        {"name": "ViT Warm KD (INT8)",      "map": 41.34, "lat_gpu": 0.70, "disk_mb": 11.26, "params_m": 11.03, "tier": "Warm KD INT8", "color": "#74c476", "marker": "p"},
        {"name": "Pruned 20% (FP16)",       "map": 43.07, "lat_gpu": 0.62, "disk_mb": 34.54, "params_m": 9.02,  "tier": "Pruned 20%", "color": "#e6550d", "marker": "*"},
        {"name": "Pruned 20% (INT8)",       "map": 42.99, "lat_gpu": 0.60, "disk_mb": 9.33,  "params_m": 9.02,  "tier": "Pruned 20% INT8", "color": "#fd8d3c", "marker": "d"},
        {"name": "Pruned 40% (FP16)",       "map": 42.28, "lat_gpu": 0.55, "disk_mb": 26.89, "params_m": 7.01,  "tier": "Pruned 40%", "color": "#d95f02", "marker": "^"},
        {"name": "Pruned 40% (INT8)",       "map": 42.27, "lat_gpu": 0.53, "disk_mb": 7.40,  "params_m": 7.01,  "tier": "Pruned 40% INT8", "color": "#fdae6b", "marker": "v"},
        {"name": "Pruned 60% (FP16)",       "map": 40.09, "lat_gpu": 0.50, "disk_mb": 16.96, "params_m": 4.41,  "tier": "Pruned 60%", "color": "#756bb1", "marker": "h"},
        {"name": "Pruned 60% (INT8)",       "map": 39.99, "lat_gpu": 0.48, "disk_mb": 4.90,  "params_m": 4.41,  "tier": "Pruned 60% INT8", "color": "#bcbddc", "marker": "8"},
    ]

    # -------------------------------------------------------------
    # Panel 1: mAP Accuracy vs. Batch-1 Hardware GPU Latency
    # -------------------------------------------------------------
    ax1 = axes[0]
    for d in data:
        size = 280 if "*" in d["marker"] else 180
        ax1.scatter(d["lat_gpu"], d["map"], color=d["color"], marker=d["marker"], s=size, edgecolors="black", linewidth=1.2, zorder=4, label=d["name"])

    # Draw Pareto Frontier curve on Panel 1
    # Frontier points: Pruned 60% INT8 (0.48ms, 39.99%) -> Pruned 60% FP16 (0.50ms, 40.09%) -> Pruned 40% INT8 (0.53ms, 42.27%) -> Pruned 40% FP16 (0.55ms, 42.28%) -> Pruned 20% INT8 (0.60ms, 42.99%) -> Pruned 20% FP16 (0.62ms, 43.07%)
    frontier_x1 = [0.48, 0.50, 0.53, 0.55, 0.60, 0.62]
    frontier_y1 = [39.99, 40.09, 42.27, 42.28, 42.99, 43.07]
    ax1.plot(frontier_x1, frontier_y1, color="#e41a1c", linestyle="--", linewidth=2.5, alpha=0.85, zorder=3, label="Optimal Pareto Frontier")
    ax1.fill_between(frontier_x1, frontier_y1, 30, color="#fee0d2", alpha=0.4, zorder=1)

    # Annotations
    ax1.annotate("Pruned 20% (Peak mAP: 43.07%)", xy=(0.62, 43.07), xytext=(0.75, 43.5),
                 arrowprops=dict(arrowstyle="->", color="#e6550d", lw=1.5), fontweight="bold", fontsize=9)
    ax1.annotate("Pruned 60% (Max Speed: 0.50ms / 1,982 FPS)", xy=(0.50, 40.09), xytext=(0.70, 39.2),
                 arrowprops=dict(arrowstyle="->", color="#756bb1", lw=1.5), fontweight="bold", fontsize=9)
    ax1.annotate("CrossViT Teacher (33.51%)", xy=(3.29, 33.51), xytext=(2.2, 35.5),
                 arrowprops=dict(arrowstyle="->", color="#7f7f7f", lw=1.5), fontweight="bold", fontsize=9)

    ax1.set_title("Perception Accuracy (mAP) vs. Batch-1 GPU Latency", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Batch-1 GPU Latency (ms) — Lower is Better", fontsize=11, fontweight="semibold")
    ax1.set_ylabel("BDD100K Multi-Label mAP (%) — Higher is Better", fontsize=11, fontweight="semibold")
    ax1.set_ylim(30, 45)
    ax1.set_xlim(0.3, 3.6)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # -------------------------------------------------------------
    # Panel 2: mAP Accuracy vs. Model Disk Footprint (MB)
    # -------------------------------------------------------------
    ax2 = axes[1]
    for d in data:
        size = 280 if "*" in d["marker"] else 180
        ax2.scatter(d["disk_mb"], d["map"], color=d["color"], marker=d["marker"], s=size, edgecolors="black", linewidth=1.2, zorder=4)

    # Draw Pareto Frontier curve on Panel 2
    # Frontier points: Pruned 60% INT8 (4.90MB, 39.99%) -> Pruned 40% INT8 (7.40MB, 42.27%) -> Pruned 20% INT8 (9.33MB, 42.99%) -> Pruned 20% FP16 (34.54MB, 43.07%)
    frontier_x2 = [4.90, 7.40, 9.33, 34.54]
    frontier_y2 = [39.99, 42.27, 42.99, 43.07]
    ax2.plot(frontier_x2, frontier_y2, color="#e41a1c", linestyle="--", linewidth=2.5, alpha=0.85, zorder=3, label="Optimal Pareto Frontier")
    ax2.fill_between(frontier_x2, frontier_y2, 30, color="#fee0d2", alpha=0.4, zorder=1)

    # Annotations
    ax2.annotate("Pruned 60% INT8 (4.90 MB, 40.0% mAP)\n44.5x Smaller than Teacher!", xy=(4.90, 39.99), xytext=(12, 38.0),
                 arrowprops=dict(arrowstyle="->", color="#bcbddc", lw=1.5), fontweight="bold", fontsize=9)
    ax2.annotate("Pruned 20% INT8 (9.33 MB, 42.99% mAP)", xy=(9.33, 42.99), xytext=(15, 43.6),
                 arrowprops=dict(arrowstyle="->", color="#fd8d3c", lw=1.5), fontweight="bold", fontsize=9)
    ax2.annotate("Teacher Baseline (109.5 MB)", xy=(109.50, 33.51), xytext=(70, 35.5),
                 arrowprops=dict(arrowstyle="->", color="#7f7f7f", lw=1.5), fontweight="bold", fontsize=9)

    ax2.set_title("Perception Accuracy (mAP) vs. Model Storage Footprint", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Model Disk Footprint (MB) — Lower is Better", fontsize=11, fontweight="semibold")
    ax2.set_ylabel("BDD100K Multi-Label mAP (%) — Higher is Better", fontsize=11, fontweight="semibold")
    ax2.set_ylim(30, 45)
    ax2.set_xlim(0, 115)
    ax2.grid(True, linestyle="--", alpha=0.6)

    # Global Title & Legend
    fig.suptitle("Vision Transformer Edge Compression Pipeline: Pareto Optimization Frontiers", fontsize=15, fontweight="bold", y=0.98)
    
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.05), frameon=True)

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Pareto Frontier plot successfully saved to: {out_path}")

if __name__ == "__main__":
    generate_pareto_plots()
