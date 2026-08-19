from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

CLASS_NAMES = [
    "bus",
    "car",
    "motorcycle",
    "truck",
]

IMAGE_SIZE = 224


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# DATASET
# ============================================================

class VehicleDataset(Dataset):

    def __init__(
        self,
        split,
        transform=None,
    ):

        self.split = split
        self.transform = transform

        self.split_dir = ROOT / split

        self.csv_path = (
            self.split_dir / "_classes.csv"
        )

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"CSV not found: {self.csv_path}"
            )

        self.data = pd.read_csv(
            self.csv_path
        )

        # Remove accidental whitespace
        self.data.columns = (
            self.data.columns
            .str.strip()
        )

        self.data["filename"] = (
            self.data["filename"]
            .astype(str)
            .str.strip()
        )

        # Verify expected columns
        required_columns = (
            ["filename"] + CLASS_NAMES
        )

        missing_columns = [
            col
            for col in required_columns
            if col not in self.data.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing columns: "
                f"{missing_columns}"
            )

        # Verify image files
        self.image_paths = [
            self.split_dir / filename
            for filename in self.data["filename"]
        ]

        missing_images = [
            str(path)
            for path in self.image_paths
            if not path.exists()
        ]

        if missing_images:
            raise FileNotFoundError(
                f"Missing {len(missing_images)} "
                f"image files."
            )

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_path = (
            self.split_dir
            / row["filename"]
        )

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")

        # ----------------------------------------------------
        # Apply transforms
        # ----------------------------------------------------

        if self.transform is not None:

            image = self.transform(
                image
            )

        # ----------------------------------------------------
        # Multi-label target
        # ----------------------------------------------------

        labels = torch.tensor(
            row[CLASS_NAMES]
            .values.astype("float32")
        )

        return image, labels


# ============================================================
# DATALOADER FACTORY
# ============================================================

def create_dataloaders(
    batch_size=32,
    num_workers=0,
):

    train_dataset = VehicleDataset(
        split="train",
        transform=train_transform,
    )

    valid_dataset = VehicleDataset(
        split="valid",
        transform=eval_transform,
    )

    test_dataset = VehicleDataset(
        split="test",
        transform=eval_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return (
        train_loader,
        valid_loader,
        test_loader,
    )


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("VEHICLE DATASET PIPELINE TEST")
    print("=" * 60)

    train_loader, valid_loader, test_loader = (
        create_dataloaders(
            batch_size=8,
            num_workers=0,
        )
    )

    print(
        f"\nTrain batches: "
        f"{len(train_loader):,}"
    )

    print(
        f"Validation batches: "
        f"{len(valid_loader):,}"
    )

    print(
        f"Test batches: "
        f"{len(test_loader):,}"
    )

    # Get one batch
    images, labels = next(
        iter(train_loader)
    )

    print("\nBATCH TEST")
    print("-" * 60)

    print(
        f"Images shape : "
        f"{images.shape}"
    )

    print(
        f"Labels shape : "
        f"{labels.shape}"
    )

    print(
        f"Image dtype  : "
        f"{images.dtype}"
    )

    print(
        f"Label dtype  : "
        f"{labels.dtype}"
    )

    print(
        f"Image range  : "
        f"{images.min():.3f} → "
        f"{images.max():.3f}"
    )

    print("\nFIRST 8 LABELS")
    print("-" * 60)

    for i, label in enumerate(labels):

        active_classes = [
            CLASS_NAMES[j]
            for j, value in enumerate(label)
            if value == 1
        ]

        print(
            f"{i}: {active_classes}"
        )

    print("\n" + "=" * 60)
    print("DATASET PIPELINE TEST COMPLETE")
    print("=" * 60)