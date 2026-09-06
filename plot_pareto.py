import os
import matplotlib.pyplot as plt
import numpy as np


def generate_pareto_plots(output_dir="assets"):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "pareto_frontier.png")

    # Set clean aesthetic styling
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

    # Benchmark dataset containing measured metrics
    # (Name, mAP %, CPU_Latency_p50_ms, Disk_Size_MB, Color, Marker)
    data = [
        {"name": "CrossViT Teacher (FP32)", "map": 33.51, "cpu_lat": 13.28, "disk_mb": 109.50, "color": "#7f7f7f", "marker": "s"},
        {"name": "CrossViT Teacher (INT8)", "map": 33.63, "cpu_lat": 12.95, "disk_mb": 30.07,  "color": "#a55194", "marker": "D"},
        {"name": "ViT Scratch Baseline",   "map": 40.52, "cpu_lat": 8.99,  "disk_mb": 42.20,  "color": "#3182bd", "marker": "o"},
        {"name": "ViT Warm KD (FP32)",      "map": 42.28, "cpu_lat": 8.99,  "disk_mb": 42.20,  "color": "#2ca02c", "marker": "P"},
        {"name": "ViT Warm KD (INT8)",      "map": 41.34, "cpu_lat": 6.67,  "disk_mb": 11.26,  "color": "#74c476", "marker": "p"},
        {"name": "Pruned 20% (FP32)",       "map": 43.07, "cpu_lat": 6.58,  "disk_mb": 34.54,  "color": "#e6550d", "marker": "*"},
        {"name": "Pruned 20% (INT8)",       "map": 42.99, "cpu_lat": 4.95,  "disk_mb": 9.33,   "color": "#fd8d3c", "marker": "d"},
        {"name": "Pruned 40% (FP32)",       "map": 42.28, "cpu_lat": 6.42,  "disk_mb": 26.89,  "color": "#d95f02", "marker": "^"},
        {"name": "Pruned 40% (INT8)",       "map": 42.27, "cpu_lat": 5.27,  "disk_mb": 7.40,   "color": "#fdae6b", "marker": "v"},
        {"name": "Pruned 60% (FP32)",       "map": 40.09, "cpu_lat": 4.63,  "disk_mb": 16.96,  "color": "#756bb1", "marker": "h"},
        {"name": "Pruned 60% (INT8)",       "map": 39.99, "cpu_lat": 3.28,  "disk_mb": 4.90,   "color": "#bcbddc", "marker": "8"},
    ]

    # =========================================================================
    # Panel 1: Perception Accuracy (mAP %) vs. CPU Edge Latency (ms)
    # =========================================================================
    ax1 = axes[0]
    for d in data:
        size = 280 if "*" in d["marker"] else 180
        ax1.scatter(d["cpu_lat"], d["map"], color=d["color"], marker=d["marker"],
                    s=size, edgecolors="black", linewidth=1.2, zorder=4, label=d["name"])

    # Optimal Pareto Frontier Curve (Lowest Latency for Given Accuracy)
    # Points on the frontier: Pruned 60% INT8 (3.28ms, 39.99%) -> Pruned 20% INT8 (4.95ms, 42.99%) -> Pruned 20% FP32 (6.58ms, 43.07%)
    frontier_x1 = [3.28, 4.95, 6.58]
    frontier_y1 = [39.99, 42.99, 43.07]
    ax1.plot(frontier_x1, frontier_y1, color="#e41a1c", linestyle="--", linewidth=2.5, alpha=0.85, zorder=3, label="Optimal Pareto Frontier")
    ax1.fill_between(frontier_x1, frontier_y1, 30, color="#fee0d2", alpha=0.35, zorder=1)

    # Annotations
    ax1.annotate("Pruned 60% INT8 (3.28ms / 305 FPS)\nFastest Edge Model", xy=(3.28, 39.99), xytext=(4.0, 37.5),
                 arrowprops=dict(arrowstyle="->", color="#756bb1", lw=1.5), fontweight="bold", fontsize=9)
    ax1.annotate("Pruned 20% INT8 (4.95ms / 42.99% mAP)\nBest Pareto Balance", xy=(4.95, 42.99), xytext=(6.0, 44.0),
                 arrowprops=dict(arrowstyle="->", color="#fd8d3c", lw=1.5), fontweight="bold", fontsize=9)
    ax1.annotate("CrossViT Teacher Baseline\n(13.28ms, 33.51% mAP)", xy=(13.28, 33.51), xytext=(10.0, 35.0),
                 arrowprops=dict(arrowstyle="->", color="#7f7f7f", lw=1.5), fontweight="bold", fontsize=9)

    ax1.set_title("Perception Accuracy (mAP) vs. CPU Latency (ms)", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Batch-1 CPU Latency (ms) — Lower is Better ➔", fontsize=11, fontweight="bold")
    ax1.set_ylabel("BDD100K Multi-Label mAP (%) — Higher is Better ➔", fontsize=11, fontweight="bold")
    ax1.set_ylim(30, 46)
    ax1.set_xlim(2.5, 14.5)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # =========================================================================
    # Panel 2: Perception Accuracy (mAP %) vs. Model Disk Footprint (MB)
    # =========================================================================
    ax2 = axes[1]
    for d in data:
        size = 280 if "*" in d["marker"] else 180
        ax2.scatter(d["disk_mb"], d["map"], color=d["color"], marker=d["marker"],
                    s=size, edgecolors="black", linewidth=1.2, zorder=4)

    # Optimal Pareto Frontier Curve for Storage
    # Frontier: Pruned 60% INT8 (4.9MB, 39.99%) -> Pruned 40% INT8 (7.4MB, 42.27%) -> Pruned 20% INT8 (9.33MB, 42.99%) -> Pruned 20% FP32 (34.54MB, 43.07%)
    frontier_x2 = [4.90, 7.40, 9.33, 34.54]
    frontier_y2 = [39.99, 42.27, 42.99, 43.07]
    ax2.plot(frontier_x2, frontier_y2, color="#e41a1c", linestyle="--", linewidth=2.5, alpha=0.85, zorder=3, label="Optimal Pareto Frontier")
    ax2.fill_between(frontier_x2, frontier_y2, 30, color="#fee0d2", alpha=0.35, zorder=1)

    # Annotations
    ax2.annotate("Pruned 60% INT8 (4.90 MB, 40.0% mAP)\n22.3x Smaller than Teacher!", xy=(4.90, 39.99), xytext=(12, 37.5),
                 arrowprops=dict(arrowstyle="->", color="#bcbddc", lw=1.5), fontweight="bold", fontsize=9)
    ax2.annotate("Pruned 20% INT8 (9.33 MB, 42.99% mAP)", xy=(9.33, 42.99), xytext=(15, 44.0),
                 arrowprops=dict(arrowstyle="->", color="#fd8d3c", lw=1.5), fontweight="bold", fontsize=9)
    ax2.annotate("Teacher Baseline (109.5 MB)", xy=(109.50, 33.51), xytext=(70, 35.5),
                 arrowprops=dict(arrowstyle="->", color="#7f7f7f", lw=1.5), fontweight="bold", fontsize=9)

    ax2.set_title("Perception Accuracy (mAP) vs. Storage Footprint (MB)", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Model Disk Footprint (MB) — Lower is Better ➔", fontsize=11, fontweight="bold")
    ax2.set_ylabel("BDD100K Multi-Label mAP (%) — Higher is Better ➔", fontsize=11, fontweight="bold")
    ax2.set_ylim(30, 46)
    ax2.set_xlim(0, 115)
    ax2.grid(True, linestyle="--", alpha=0.6)

    # Global Title & Legend
    fig.suptitle("Vision Transformer Edge Compression: Multi-Objective Pareto Frontiers", fontsize=15, fontweight="bold", y=0.98)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.06), frameon=True)

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Pareto Frontier plot successfully generated and saved to: {out_path}")


if __name__ == "__main__":
    generate_pareto_plots()

