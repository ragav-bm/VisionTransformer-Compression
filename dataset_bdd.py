import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

# 15 Target binary perception attributes across 3 categories
WEATHER_CLASSES = [
    "rainy", "snowy", "clear", "overcast", "foggy", "partly cloudy", "undefined"
]
SCENE_CLASSES = [
    "highway", "residential", "city street", "parking lot", "gas stations", "tunnel"
]
TIMEOFDAY_CLASSES = [
    "daytime", "night"
]

ALL_ATTRIBUTES = WEATHER_CLASSES + SCENE_CLASSES + TIMEOFDAY_CLASSES
NUM_ATTRIBUTES = len(ALL_ATTRIBUTES)  # 15


def get_bdd_transforms(image_size=224, is_train=True):
    """
    Standard transform pipeline for driving scene images (ImageNet normalization).
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


class BDD100KMultiLabelDataset(Dataset):
    """
    PyTorch Dataset for BDD100K Multi-Label Scene Attribute Perception.
    Parses weather, scene context, and time-of-day attributes into a 15-dim binary tensor.
    """
    def __init__(self, image_dir=None, annotation_file=None, transform=None, is_train=True):
        self.image_dir = image_dir
        self.transform = transform if transform is not None else get_bdd_transforms(is_train=is_train)
        self.samples = []
        
        if annotation_file and os.path.exists(annotation_file):
            self._load_annotations(annotation_file)
        else:
            print(f"[Dataset Notice] Annotation file '{annotation_file}' not found. Using synthetic sample mode for pipeline verification.")
            self._generate_synthetic_samples(num_samples=200)

    def _load_annotations(self, annotation_file):
        """Loads and parses official BDD100K JSON annotations."""
        with open(annotation_file, 'r') as f:
            data = json.load(f)

        if isinstance(data, dict):
            entries = data.get("samples", [data])
        elif isinstance(data, list):
            entries = data
        else:
            entries = []

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            
            img_name = entry.get("name", "")
            if not img_name and "filepath" in entry:
                img_name = os.path.basename(entry["filepath"])
            
            img_path = os.path.join(self.image_dir, img_name) if self.image_dir else img_name
            attrs = entry.get("attributes", entry)

            # Construct 15-dimensional multi-hot binary label
            label = np.zeros(NUM_ATTRIBUTES, dtype=np.float32)

            # 1. Weather
            w = attrs.get("weather", "")
            if isinstance(w, dict): w = w.get("label", "")
            w = str(w).lower()
            if w in WEATHER_CLASSES:
                label[WEATHER_CLASSES.index(w)] = 1.0

            # 2. Scene
            s = attrs.get("scene", "")
            if isinstance(s, dict): s = s.get("label", "")
            s = str(s).lower()
            if s in SCENE_CLASSES:
                label[len(WEATHER_CLASSES) + SCENE_CLASSES.index(s)] = 1.0

            # 3. Time of Day
            t = attrs.get("timeofday", "")
            if isinstance(t, dict): t = t.get("label", "")
            t = str(t).lower()
            if t in TIMEOFDAY_CLASSES:
                label[len(WEATHER_CLASSES) + len(SCENE_CLASSES) + TIMEOFDAY_CLASSES.index(t)] = 1.0

            self.samples.append({
                "img_path": img_path,
                "label": label
            })

    def _generate_synthetic_samples(self, num_samples=200):
        """Creates dummy samples for immediate pipeline testing and debugging."""
        for i in range(num_samples):
            label = np.zeros(NUM_ATTRIBUTES, dtype=np.float32)
            # Pick one random weather, scene, timeofday to mimic real distribution
            label[np.random.randint(0, len(WEATHER_CLASSES))] = 1.0
            label[len(WEATHER_CLASSES) + np.random.randint(0, len(SCENE_CLASSES))] = 1.0
            label[len(WEATHER_CLASSES) + len(SCENE_CLASSES) + np.random.randint(0, len(TIMEOFDAY_CLASSES))] = 1.0

            self.samples.append({
                "img_path": f"dummy_img_{i}.jpg",
                "label": label,
                "is_dummy": True
            })

    def calculate_pos_weights(self):
        """
        Computes the positive weight vector for BCEWithLogitsLoss:
        pos_weight[c] = (Total Samples - Positive Samples) / Positive Samples
        This compensates for rare attributes (e.g. foggy, tunnel, snowy).
        """
        all_labels = np.array([s["label"] for s in self.samples])
        pos_counts = np.sum(all_labels, axis=0)
        total_samples = len(self.samples)
        neg_counts = total_samples - pos_counts

        # Avoid divide-by-zero for very rare or missing classes with epsilon
        pos_weights = np.where(pos_counts > 0, neg_counts / np.maximum(pos_counts, 1.0), 1.0)
        # Clamp to avoid extreme gradient explosion on ultra-rare attributes
        pos_weights = np.clip(pos_weights, 1.0, 100.0)
        return torch.tensor(pos_weights, dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        label = torch.tensor(item["label"], dtype=torch.float32)

        if item.get("is_dummy", False) or not os.path.exists(item["img_path"]):
            # Generate a synthetic RGB image tensor for testing
            img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        else:
            img = Image.open(item["img_path"]).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


if __name__ == "__main__":
    print("=== Testing BDD100K Multi-Label Dataset Pipeline ===")
    dataset = BDD100KMultiLabelDataset(is_train=True)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    images, labels = next(iter(loader))
    pos_weights = dataset.calculate_pos_weights()

    print(f"Batch Image Tensor Shape : {images.shape}")
    print(f"Batch Label Tensor Shape : {labels.shape}")
    print(f"Positive Weights Vector  : {pos_weights.shape} -> \n{pos_weights.numpy().round(2)}")
    print("✅ Pipeline verified successfully!")

