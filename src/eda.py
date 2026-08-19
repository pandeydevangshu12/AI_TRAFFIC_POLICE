from pathlib import Path
from collections import Counter
import hashlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

TRAIN_DIR = ROOT / "train"
VALID_DIR = ROOT / "valid"
TEST_DIR = ROOT / "test"

OUTPUT_DIR = ROOT / "outputs" / "eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["bus", "car", "motorcycle", "truck"]

SAMPLE_SIZE = 1000
RANDOM_STATE = 42


# ============================================================
# 2. LOAD DATA
# ============================================================

def load_split(split_dir):
    csv_path = split_dir / "_classes.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Remove accidental whitespace
    df.columns = df.columns.str.strip()

    # Remove whitespace from string columns
    df["filename"] = df["filename"].astype(str).str.strip()

    return df


train_df = load_split(TRAIN_DIR)
valid_df = load_split(VALID_DIR)
test_df = load_split(TEST_DIR)


# ============================================================
# 3. BASIC DATASET INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("AI TRAFFIC POLICE - DATASET EDA")
print("=" * 70)

print("\nDATASET LOCATION")
print("-" * 70)
print(f"Root: {ROOT}")

print("\nDATASET SIZE")
print("-" * 70)

print(f"Train      : {len(train_df):,}")
print(f"Validation : {len(valid_df):,}")
print(f"Test       : {len(test_df):,}")

total_images = (
    len(train_df)
    + len(valid_df)
    + len(test_df)
)

print(f"Total      : {total_images:,}")


# ============================================================
# 4. DATAFRAME STRUCTURE
# ============================================================

print("\nDATASET COLUMNS")
print("-" * 70)

print(train_df.columns.tolist())

print("\nFIRST 5 TRAINING SAMPLES")
print("-" * 70)

print(train_df.head())


# ============================================================
# 5. VALIDATE LABEL COLUMNS
# ============================================================

print("\nLABEL VALIDATION")
print("-" * 70)

for split_name, df in {
    "Train": train_df,
    "Validation": valid_df,
    "Test": test_df
}.items():

    missing_columns = [
        col for col in CLASS_NAMES
        if col not in df.columns
    ]

    if missing_columns:
        print(
            f"{split_name}: Missing columns -> "
            f"{missing_columns}"
        )
    else:
        print(
            f"{split_name}: All label columns present ✓"
        )


# ============================================================
# 6. MISSING VALUES
# ============================================================

print("\nMISSING VALUES")
print("-" * 70)

for split_name, df in {
    "Train": train_df,
    "Validation": valid_df,
    "Test": test_df
}.items():

    missing = df.isnull().sum()

    print(f"\n{split_name}")
    print(missing)


# ============================================================
# 7. CLASS DISTRIBUTION
# ============================================================

print("\nCLASS DISTRIBUTION - TRAIN")
print("-" * 70)

train_class_counts = (
    train_df[CLASS_NAMES]
    .sum()
    .sort_values(ascending=False)
)

print(train_class_counts)

print("\nCLASS PREVALENCE (%)")
print("-" * 70)

train_class_percentage = (
    train_df[CLASS_NAMES]
    .mean()
    * 100
)

print(train_class_percentage.round(2))


# ============================================================
# 8. CLASS DISTRIBUTION PLOT
# ============================================================

plt.figure(figsize=(9, 6))

sns.barplot(
    x=train_class_counts.index,
    y=train_class_counts.values
)

plt.title("Training Set - Vehicle Class Distribution")
plt.xlabel("Vehicle Class")
plt.ylabel("Number of Images")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "class_distribution.png",
    dpi=200
)

plt.close()


# ============================================================
# 9. LABELS PER IMAGE
# ============================================================

train_df["num_labels"] = (
    train_df[CLASS_NAMES]
    .sum(axis=1)
)

valid_df["num_labels"] = (
    valid_df[CLASS_NAMES]
    .sum(axis=1)
)

test_df["num_labels"] = (
    test_df[CLASS_NAMES]
    .sum(axis=1)
)


print("\nNUMBER OF LABELS PER IMAGE - TRAIN")
print("-" * 70)

label_count_distribution = (
    train_df["num_labels"]
    .value_counts()
    .sort_index()
)

print(label_count_distribution)


plt.figure(figsize=(8, 5))

sns.barplot(
    x=label_count_distribution.index.astype(str),
    y=label_count_distribution.values
)

plt.title("Number of Vehicle Classes per Image")
plt.xlabel("Number of Labels")
plt.ylabel("Number of Images")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "labels_per_image.png",
    dpi=200
)

plt.close()


# ============================================================
# 10. LABEL COMBINATIONS
# ============================================================

def get_label_combination(row):
    labels = [
        cls
        for cls in CLASS_NAMES
        if row[cls] == 1
    ]

    if not labels:
        return "No label"

    return " + ".join(labels)


train_df["label_combination"] = train_df.apply(
    get_label_combination,
    axis=1
)

valid_df["label_combination"] = valid_df.apply(
    get_label_combination,
    axis=1
)

test_df["label_combination"] = test_df.apply(
    get_label_combination,
    axis=1
)


combination_counts = (
    train_df["label_combination"]
    .value_counts()
)


print("\nTOP 15 LABEL COMBINATIONS - TRAIN")
print("-" * 70)

print(combination_counts.head(15))


top_combinations = combination_counts.head(15)

plt.figure(figsize=(12, 7))

sns.barplot(
    x=top_combinations.values,
    y=top_combinations.index
)

plt.title("Top Vehicle-Class Combinations")
plt.xlabel("Number of Images")
plt.ylabel("Combination")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "label_combinations.png",
    dpi=200
)

plt.close()


# ============================================================
# 11. CLASS CO-OCCURRENCE
# ============================================================

co_occurrence = (
    train_df[CLASS_NAMES].T
    @ train_df[CLASS_NAMES]
)

print("\nCLASS CO-OCCURRENCE")
print("-" * 70)

print(co_occurrence)


plt.figure(figsize=(7, 6))

sns.heatmap(
    co_occurrence,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Vehicle Class Co-occurrence")
plt.xlabel("Vehicle Class")
plt.ylabel("Vehicle Class")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "class_cooccurrence.png",
    dpi=200
)

plt.close()


# ============================================================
# 12. LABEL VALIDITY
# ============================================================

print("\nLABEL VALUES")
print("-" * 70)

for cls in CLASS_NAMES:

    values = sorted(
        train_df[cls]
        .dropna()
        .unique()
        .tolist()
    )

    print(f"{cls:12}: {values}")


# ============================================================
# 13. DUPLICATE FILENAMES
# ============================================================

print("\nDUPLICATE FILENAMES")
print("-" * 70)

for split_name, df in {
    "Train": train_df,
    "Validation": valid_df,
    "Test": test_df
}.items():

    duplicates = df["filename"].duplicated().sum()

    print(
        f"{split_name:12}: "
        f"{duplicates}"
    )


# ============================================================
# 14. VERIFY IMAGE FILES
# ============================================================

def check_missing_images(df, split_dir):

    missing = []

    for filename in df["filename"]:

        image_path = split_dir / filename

        if not image_path.exists():
            missing.append(filename)

    return missing


print("\nIMAGE FILE VALIDATION")
print("-" * 70)

missing_train = check_missing_images(
    train_df,
    TRAIN_DIR
)

missing_valid = check_missing_images(
    valid_df,
    VALID_DIR
)

missing_test = check_missing_images(
    test_df,
    TEST_DIR
)

print(f"Train missing      : {len(missing_train)}")
print(f"Validation missing : {len(missing_valid)}")
print(f"Test missing       : {len(missing_test)}")


# ============================================================
# 15. IMAGE DIMENSIONS
# ============================================================

def get_dimensions(df, split_dir, sample_size=1000):

    sample = df.sample(
        min(sample_size, len(df)),
        random_state=RANDOM_STATE
    )

    dimensions = []
    corrupted = []

    for filename in sample["filename"]:

        path = split_dir / filename

        try:

            with Image.open(path) as img:

                dimensions.append(
                    img.size
                )

        except Exception:

            corrupted.append(filename)

    return dimensions, corrupted


print("\nIMAGE DIMENSIONS")
print("-" * 70)

train_dimensions, dimension_errors = get_dimensions(
    train_df,
    TRAIN_DIR,
    SAMPLE_SIZE
)

dimension_counts = Counter(
    train_dimensions
)

print(
    f"Images sampled: "
    f"{len(train_dimensions):,}"
)

print("\nMost common dimensions:")

for dimension, count in dimension_counts.most_common(15):

    print(
        f"{str(dimension):15} "
        f"{count}"
    )


if train_dimensions:

    widths = [
        dimension[0]
        for dimension in train_dimensions
    ]

    heights = [
        dimension[1]
        for dimension in train_dimensions
    ]

    print("\nWidth statistics")
    print("-" * 40)

    print(f"Min  : {min(widths)}")
    print(f"Max  : {max(widths)}")
    print(f"Mean : {np.mean(widths):.2f}")

    print("\nHeight statistics")
    print("-" * 40)

    print(f"Min  : {min(heights)}")
    print(f"Max  : {max(heights)}")
    print(f"Mean : {np.mean(heights):.2f}")


# ============================================================
# 16. WIDTH DISTRIBUTION
# ============================================================

if train_dimensions:

    plt.figure(figsize=(9, 5))

    plt.hist(
        widths,
        bins=30
    )

    plt.title("Image Width Distribution")
    plt.xlabel("Width")
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "width_distribution.png",
        dpi=200
    )

    plt.close()


# ============================================================
# 17. HEIGHT DISTRIBUTION
# ============================================================

if train_dimensions:

    plt.figure(figsize=(9, 5))

    plt.hist(
        heights,
        bins=30
    )

    plt.title("Image Height Distribution")
    plt.xlabel("Height")
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "height_distribution.png",
        dpi=200
    )

    plt.close()


# ============================================================
# 18. CORRUPTED IMAGE CHECK
# ============================================================

def find_corrupted_images(df, split_dir):

    corrupted = []

    for filename in df["filename"]:

        path = split_dir / filename

        try:

            with Image.open(path) as img:
                img.verify()

        except Exception:

            corrupted.append(filename)

    return corrupted


print("\nCORRUPTED IMAGE CHECK")
print("-" * 70)

corrupt_train = find_corrupted_images(
    train_df,
    TRAIN_DIR
)

corrupt_valid = find_corrupted_images(
    valid_df,
    VALID_DIR
)

corrupt_test = find_corrupted_images(
    test_df,
    TEST_DIR
)

print(f"Train      : {len(corrupt_train)}")
print(f"Validation : {len(corrupt_valid)}")
print(f"Test       : {len(corrupt_test)}")


# ============================================================
# 19. TRAIN / VALID / TEST DISTRIBUTION
# ============================================================

distribution = pd.DataFrame({

    "Train": train_df[CLASS_NAMES].mean(),

    "Validation": valid_df[CLASS_NAMES].mean(),

    "Test": test_df[CLASS_NAMES].mean()

}) * 100


print("\nCLASS DISTRIBUTION ACROSS SPLITS (%)")
print("-" * 70)

print(
    distribution.round(2)
)


distribution.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title(
    "Class Distribution Across Dataset Splits"
)

plt.xlabel("Vehicle Class")

plt.ylabel(
    "Percentage of Images (%)"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "split_distribution.png",
    dpi=200
)

plt.close()


# ============================================================
# 20. SAMPLE IMAGES
# ============================================================

def save_sample_images(
    df,
    split_dir,
    output_path,
    n=12
):

    samples = df.sample(
        min(n, len(df)),
        random_state=RANDOM_STATE
    )

    rows = 3
    cols = 4

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(16, 11)
    )

    axes = axes.flatten()

    for ax in axes:
        ax.axis("off")

    for ax, (_, row) in zip(
        axes,
        samples.iterrows()
    ):

        path = split_dir / row["filename"]

        try:

            image = Image.open(path)

            labels = [
                cls
                for cls in CLASS_NAMES
                if row[cls] == 1
            ]

            ax.imshow(image)

            ax.set_title(
                ", ".join(labels),
                fontsize=10
            )

            ax.axis("off")

        except Exception as e:

            ax.set_title(
                f"Error: {e}"
            )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200
    )

    plt.close()


print("\nGENERATING SAMPLE IMAGE GRID...")

save_sample_images(
    train_df,
    TRAIN_DIR,
    OUTPUT_DIR / "sample_images.png"
)

print(
    f"Saved: "
    f"{OUTPUT_DIR / 'sample_images.png'}"
)


# ============================================================
# 21. SAMPLES PER CLASS
# ============================================================

print("\nGENERATING CLASS SAMPLE GRIDS...")
print("-" * 70)

for cls in CLASS_NAMES:

    subset = train_df[
        train_df[cls] == 1
    ]

    if len(subset) == 0:
        continue

    output_path = (
        OUTPUT_DIR
        / f"samples_{cls}.png"
    )

    save_sample_images(
        subset,
        TRAIN_DIR,
        output_path,
        n=12
    )

    print(
        f"{cls:12}: "
        f"{len(subset):,} images"
    )


# ============================================================
# 22. CROSS-SPLIT DUPLICATE CHECK
# ============================================================

print("\nCROSS-SPLIT FILENAME OVERLAP")
print("-" * 70)

train_files = set(
    train_df["filename"]
)

valid_files = set(
    valid_df["filename"]
)

test_files = set(
    test_df["filename"]
)

print(
    "Train ∩ Valid:",
    len(train_files & valid_files)
)

print(
    "Train ∩ Test :",
    len(train_files & test_files)
)

print(
    "Valid ∩ Test :",
    len(valid_files & test_files)
)


# ============================================================
# 23. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL EDA SUMMARY")
print("=" * 70)

print(f"""
Dataset
-------
Train images       : {len(train_df):,}
Validation images  : {len(valid_df):,}
Test images        : {len(test_df):,}
Total images       : {total_images:,}

Classes
-------
{CLASS_NAMES}

Train class counts
------------------
{train_class_counts.to_string()}

Labels per image
----------------
{label_count_distribution.to_string()}

Image validation
----------------
Missing train      : {len(missing_train)}
Missing validation : {len(missing_valid)}
Missing test       : {len(missing_test)}

Corrupted images
----------------
Train      : {len(corrupt_train)}
Validation : {len(corrupt_valid)}
Test       : {len(corrupt_test)}

EDA outputs
-----------
{OUTPUT_DIR}
""")

print("=" * 70)
print("EDA COMPLETE")
print("=" * 70)