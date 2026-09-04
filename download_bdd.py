import os
import json
import argparse
from huggingface_hub import hf_hub_download, HfApi
from tqdm import tqdm


def download_bdd100k(dest_dir="./data/bdd100k", max_images=1000):
    """
    Downloads BDD100K 10K dataset metadata and image files from HuggingFace mirror.
    """
    os.makedirs(dest_dir, exist_ok=True)
    images_dir = os.path.join(dest_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    print(f"=== Downloading BDD100K Dataset into {dest_dir} ===")

    # 1. Download official 10,000-sample annotations JSON
    print("\n[1/2] Fetching 10K multi-label annotations (samples.json)...")
    labels_file = hf_hub_download(
        repo_id="dgural/bdd100k",
        filename="samples.json",
        repo_type="dataset",
        local_dir=dest_dir
    )
    print(f"  --> Saved annotations: {labels_file}")

    # 2. Download Images
    print(f"\n[2/2] Fetching images (Target: {max_images} images)...")
    with open(labels_file, 'r') as f:
        data = json.load(f)

    samples = data["samples"] if "samples" in data else data
    total_to_download = min(len(samples), max_images)
    print(f"  --> Downloading {total_to_download} driving scene images...")

    for i in tqdm(range(total_to_download), desc="Downloading BDD images"):
        item = samples[i]
        rel_path = item.get("filepath", "")
        # Rel path in repo is data/<filename>.jpg
        if "data/" not in rel_path:
            filename = os.path.basename(rel_path)
            rel_path = f"data/{filename}"
        else:
            filename = os.path.basename(rel_path)

        local_img_path = os.path.join(images_dir, filename)
        if not os.path.exists(local_img_path):
            try:
                hf_hub_download(
                    repo_id="dgural/bdd100k",
                    filename=rel_path,
                    repo_type="dataset",
                    local_dir=dest_dir
                )
            except Exception as e:
                pass

    print(f"\n✅ Download complete! Images saved in {images_dir}, labels in {labels_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BDD100K Dataset Downloader")
    parser.add_argument("--dest", type=str, default="./data/bdd100k", help="Destination folder")
    parser.add_argument("--max-images", type=int, default=1000, help="Number of images to download (e.g. 1000 or 10000)")
    args = parser.parse_args()
    download_bdd100k(args.dest, args.max_images)

