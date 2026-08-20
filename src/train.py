import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.optim import Adam

from dataset import create_dataloaders
from model import VehicleCNN

import os

CPU_THREADS = os.cpu_count()

torch.set_num_threads(CPU_THREADS)
torch.set_num_interop_threads(max(1, CPU_THREADS // 2))

print(f"CPU threads: {CPU_THREADS}")


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = ROOT / "outputs" / "training"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "best_model.pth"
HISTORY_PATH = OUTPUT_DIR / "training_history.json"

CLASS_NAMES = [
    "bus",
    "car",
    "motorcycle",
    "truck",
]

NUM_CLASSES = len(CLASS_NAMES)

BATCH_SIZE = 32
NUM_WORKERS = 0

EPOCHS = 15

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

THRESHOLD = 0.5

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("AI TRAFFIC POLICE - BASELINE TRAINING")
print("=" * 70)

print(f"\nDevice: {DEVICE}")

if DEVICE.type == "cuda":
    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )
else:
    print("GPU: None")
    print("Running on CPU.")


# ============================================================
# DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATA")
print("=" * 70)

train_loader, valid_loader, test_loader = (
    create_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
)

print(
    f"Train batches      : "
    f"{len(train_loader):,}"
)

print(
    f"Validation batches : "
    f"{len(valid_loader):,}"
)

print(
    f"Test batches       : "
    f"{len(test_loader):,}"
)


# ============================================================
# MODEL
# ============================================================

model = VehicleCNN(
    num_classes=NUM_CLASSES
)

model = model.to(DEVICE)


total_parameters = sum(
    p.numel()
    for p in model.parameters()
)

print("\n" + "=" * 70)
print("MODEL")
print("=" * 70)

print(
    f"Parameters: "
    f"{total_parameters:,}"
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.BCEWithLogitsLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    all_targets,
    all_probabilities,
    threshold=0.5,
):

    predictions = (
        all_probabilities >= threshold
    ).astype(int)

    targets = all_targets.astype(int)

    precision = precision_score(
        targets,
        predictions,
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
        targets,
        predictions,
        average="macro",
        zero_division=0,
    )

    f1 = f1_score(
        targets,
        predictions,
        average="macro",
        zero_division=0,
    )

    per_class_f1 = f1_score(
        targets,
        predictions,
        average=None,
        zero_division=0,
    )

    return (
        precision,
        recall,
        f1,
        per_class_f1,
    )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):

    model.train()

    running_loss = 0.0

    total_samples = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        # ----------------------------------------------------
        # Clear old gradients
        # ----------------------------------------------------

        optimizer.zero_grad()

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        logits = model(images)

        # ----------------------------------------------------
        # Calculate loss
        # ----------------------------------------------------

        loss = criterion(
            logits,
            labels,
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()

        # ----------------------------------------------------
        # Update weights
        # ----------------------------------------------------

        optimizer.step()

        batch_size = images.size(0)

        running_loss += (
            loss.item()
            * batch_size
        )

        total_samples += batch_size

    epoch_loss = (
        running_loss
        /
        total_samples
    )

    return epoch_loss


# ============================================================
# VALIDATION
# ============================================================

def evaluate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    running_loss = 0.0

    total_samples = 0

    all_targets = []
    all_probabilities = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            logits = model(images)

            # ------------------------------------------------
            # Loss
            # ------------------------------------------------

            loss = criterion(
                logits,
                labels,
            )

            batch_size = images.size(0)

            running_loss += (
                loss.item()
                * batch_size
            )

            total_samples += batch_size

            # ------------------------------------------------
            # Convert logits → probabilities
            # ------------------------------------------------

            probabilities = torch.sigmoid(
                logits
            )

            all_targets.append(
                labels.cpu().numpy()
            )

            all_probabilities.append(
                probabilities.cpu().numpy()
            )

    epoch_loss = (
        running_loss
        /
        total_samples
    )

    all_targets = np.concatenate(
        all_targets,
        axis=0,
    )

    all_probabilities = np.concatenate(
        all_probabilities,
        axis=0,
    )

    (
        precision,
        recall,
        f1,
        per_class_f1,
    ) = calculate_metrics(
        all_targets,
        all_probabilities,
        threshold=THRESHOLD,
    )

    return (
        epoch_loss,
        precision,
        recall,
        f1,
        per_class_f1,
    )


# ============================================================
# TRAINING LOOP
# ============================================================

history = {
    "train_loss": [],
    "val_loss": [],
    "val_precision": [],
    "val_recall": [],
    "val_f1": [],
    "val_f1_bus": [],
    "val_f1_car": [],
    "val_f1_motorcycle": [],
    "val_f1_truck": [],
}

best_val_f1 = -1.0


print("\n" + "=" * 70)
print("TRAINING")
print("=" * 70)

for epoch in range(1, EPOCHS + 1):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train_loss = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        DEVICE,
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    (
        val_loss,
        val_precision,
        val_recall,
        val_f1,
        per_class_f1,
    ) = evaluate(
        model,
        valid_loader,
        criterion,
        DEVICE,
    )

    # --------------------------------------------------------
    # Store history
    # --------------------------------------------------------

    history["train_loss"].append(
        train_loss
    )

    history["val_loss"].append(
        val_loss
    )

    history["val_precision"].append(
        val_precision
    )

    history["val_recall"].append(
        val_recall
    )

    history["val_f1"].append(
        val_f1
    )

    for class_name, class_f1 in zip(
        CLASS_NAMES,
        per_class_f1,
    ):

        history[
            f"val_f1_{class_name}"
        ].append(
            float(class_f1)
        )

    # --------------------------------------------------------
    # Print epoch results
    # --------------------------------------------------------

    print(
        f"\nEpoch "
        f"{epoch:02d}/{EPOCHS}"
    )

    print(
        f"Train Loss : "
        f"{train_loss:.4f}"
    )

    print(
        f"Val Loss   : "
        f"{val_loss:.4f}"
    )

    print(
        f"Precision  : "
        f"{val_precision:.4f}"
    )

    print(
        f"Recall     : "
        f"{val_recall:.4f}"
    )

    print(
        f"F1         : "
        f"{val_f1:.4f}"
    )

    print("Per-class F1:")

    for class_name, class_f1 in zip(
        CLASS_NAMES,
        per_class_f1,
    ):

        print(
            f"  {class_name:12}: "
            f"{class_f1:.4f}"
        )

    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if val_f1 > best_val_f1:

        best_val_f1 = val_f1

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "val_f1":
                    best_val_f1,

                "epoch":
                    epoch,

                "class_names":
                    CLASS_NAMES,

                "threshold":
                    THRESHOLD,
            },
            MODEL_PATH,
        )

        print(
            "✓ New best model saved"
        )


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

with open(
    HISTORY_PATH,
    "w",
) as f:

    json.dump(
        history,
        f,
        indent=4,
    )


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

print(
    f"Best epoch : "
    f"{checkpoint['epoch']}"
)

print(
    f"Best Val F1: "
    f"{checkpoint['val_f1']:.4f}"
)


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

(
    test_loss,
    test_precision,
    test_recall,
    test_f1,
    test_per_class_f1,
) = evaluate(
    model,
    test_loader,
    criterion,
    DEVICE,
)

print(
    f"\nTest Loss      : "
    f"{test_loss:.4f}"
)

print(
    f"Test Precision : "
    f"{test_precision:.4f}"
)

print(
    f"Test Recall    : "
    f"{test_recall:.4f}"
)

print(
    f"Test F1        : "
    f"{test_f1:.4f}"
)

print("\nTest Per-class F1:")

for class_name, class_f1 in zip(
    CLASS_NAMES,
    test_per_class_f1,
):

    print(
        f"  {class_name:12}: "
        f"{class_f1:.4f}"
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    f"\nBest model:\n"
    f"{MODEL_PATH}"
)

print(
    f"\nTraining history:\n"
    f"{HISTORY_PATH}"
)

print(
    f"\nBest validation F1: "
    f"{best_val_f1:.4f}"
)

print(
    f"Final test F1: "
    f"{test_f1:.4f}"
)