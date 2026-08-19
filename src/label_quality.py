from pathlib import Path
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image, ImageStat
import hashlib


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

TRAIN_DIR = ROOT / "train"
VALID_DIR = ROOT / "valid"
TEST_DIR = ROOT / "test"

OUTPUT_DIR = ROOT / "outputs" / "label_quality"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "bus",
    "car",
    "motorcycle",
    "truck"
]

RANDOM_STATE = 42

# Number of images to inspect visually
SAMPLE_SIZE = 500


# ============================================================
# LOAD DATA
# ============================================================

def load_split(split_dir):

    csv_path = split_dir / "_classes.csv"

    df = pd.read_csv(csv_path)

    df.columns = df.columns.str.strip()

    df["filename"] = (
        df["filename"]
        .astype(str)
        .str.strip()
    )

    return df


train_df = load_split(TRAIN_DIR)
valid_df = load_split(VALID_DIR)
test_df = load_split(TEST_DIR)


# ============================================================
# BASIC LABEL INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("LABEL QUALITY / DATASET QUALITY ANALYSIS")
print("=" * 70)


for name, df in {
    "Train": train_df,
    "Validation": valid_df,
    "Test": test_df
}.items():

    print(
        f"{name:12}: "
        f"{len(df):,} images"
    )


# ============================================================
# NUMBER OF LABELS PER IMAGE
# ============================================================

for df in [
    train_df,
    valid_df,
    test_df
]:

    df["num_labels"] = (
        df[CLASS_NAMES]
        .sum(axis=1)
    )


print("\n" + "=" * 70)
print("LABEL COUNT PER IMAGE")
print("=" * 70)


for name, df in {
    "Train": train_df,
    "Validation": valid_df,
    "Test": test_df
}.items():

    distribution = (
        df["num_labels"]
        .value_counts()
        .sort_index()
    )

    print(f"\n{name}")

    for count, number in distribution.items():

        percentage = (
            number / len(df) * 100
        )

        print(
            f"  {count} labels : "
            f"{number:,} "
            f"({percentage:.2f}%)"
        )


# ============================================================
# MULTI-LABEL PERCENTAGE
# ============================================================

print("\n" + "=" * 70)
print("MULTI-LABEL ANALYSIS")
print("=" * 70)


for name, df in {
    "Train": train_df,
    "Validation": valid_df,
    "Test": test_df
}.items():

    multi = (
        df["num_labels"] > 1
    ).sum()

    percentage = (
        multi / len(df) * 100
    )

    print(
        f"{name:12}: "
        f"{multi:,} multi-label images "
        f"({percentage:.2f}%)"
    )


# ============================================================
# LABEL COMBINATIONS
# ============================================================

def label_combination(row):

    labels = [
        cls
        for cls in CLASS_NAMES
        if row[cls] == 1
    ]

    return " + ".join(labels)


for df in [
    train_df,
    valid_df,
    test_df
]:

    df["label_combination"] = df.apply(
        label_combination,
        axis=1
    )


print("\n" + "=" * 70)
print("LABEL COMBINATIONS")
print("=" * 70)

combination_counts = (
    train_df["label_combination"]
    .value_counts()
)

print(
    combination_counts
)


# ============================================================
# RARE LABEL COMBINATIONS
# ============================================================

print("\n" + "=" * 70)
print("RARE LABEL COMBINATIONS")
print("=" * 70)

rare_combinations = (
    combination_counts[
        combination_counts <= 25
    ]
)

print(
    rare_combinations
)


# ============================================================
# CLASS BALANCE
# ============================================================

print("\n" + "=" * 70)
print("CLASS BALANCE")
print("=" * 70)

class_counts = (
    train_df[CLASS_NAMES]
    .sum()
    .sort_values(ascending=False)
)

class_percentages = (
    train_df[CLASS_NAMES]
    .mean()
    * 100
)

for cls in class_counts.index:

    print(
        f"{cls:12}: "
        f"{class_counts[cls]:,} "
        f"({class_percentages[cls]:.2f}%)"
    )


imbalance_ratio = (
    class_counts.max()
    /
    class_counts.min()
)

print(
    f"\nMax / Min class ratio: "
    f"{imbalance_ratio:.2f}"
)


# ============================================================
# CO-OCCURRENCE
# ============================================================

print("\n" + "=" * 70)
print("CLASS CO-OCCURRENCE")
print("=" * 70)

co_occurrence = (
    train_df[CLASS_NAMES].T
    @ train_df[CLASS_NAMES]
)

print(
    co_occurrence
)


# ============================================================
# CONDITIONAL CO-OCCURRENCE
# ============================================================

print("\n" + "=" * 70)
print("CONDITIONAL CO-OCCURRENCE")
print("=" * 70)

print(
    "P(B | A): probability that B appears "
    "when A is present."
)

conditional = pd.DataFrame(
    index=CLASS_NAMES,
    columns=CLASS_NAMES,
    dtype=float
)

for a in CLASS_NAMES:

    for b in CLASS_NAMES:

        if a == b:

            conditional.loc[a, b] = 1.0

        else:

            a_present = (
                train_df[a] == 1
            )

            both_present = (
                (train_df[a] == 1)
                &
                (train_df[b] == 1)
            )

            if a_present.sum() > 0:

                conditional.loc[a, b] = (
                    both_present.sum()
                    /
                    a_present.sum()
                )

            else:

                conditional.loc[a, b] = 0


print(
    conditional.round(3)
)


# ============================================================
# SAVE CONDITIONAL CO-OCCURRENCE HEATMAP
# ============================================================

plt.figure(
    figsize=(8, 6)
)

sns.heatmap(
    conditional,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    vmin=0,
    vmax=1
)

plt.title(
    "Conditional Class Co-occurrence"
)

plt.xlabel(
    "Class B"
)

plt.ylabel(
    "Class A"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "conditional_cooccurrence.png",
    dpi=200
)

plt.close()


# ============================================================
# LABEL DISTRIBUTION PLOT
# ============================================================

plt.figure(
    figsize=(8, 5)
)

sns.countplot(
    data=train_df,
    x="num_labels"
)

plt.title(
    "Number of Labels per Image"
)

plt.xlabel(
    "Number of Labels"
)

plt.ylabel(
    "Number of Images"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "label_count_distribution.png",
    dpi=200
)

plt.close()


# ============================================================
# IMAGE STATISTICS
# ============================================================

def image_statistics(
    df,
    split_dir,
    sample_size=SAMPLE_SIZE
):

    sample = df.sample(
        min(sample_size, len(df)),
        random_state=RANDOM_STATE
    )

    records = []

    for _, row in sample.iterrows():

        path = split_dir / row["filename"]

        try:

            with Image.open(path) as img:

                rgb = img.convert("RGB")

                stat = ImageStat.Stat(rgb)

                mean_brightness = (
                    sum(stat.mean) / 3
                )

                mean_contrast = (
                    sum(stat.stddev) / 3
                )

                width, height = img.size

                records.append({
                    "filename": row["filename"],
                    "brightness": mean_brightness,
                    "contrast": mean_contrast,
                    "width": width,
                    "height": height,
                    "num_labels": row["num_labels"],
                    "label_combination":
                        row["label_combination"]
                })

        except Exception:

            pass

    return pd.DataFrame(records)


print("\n" + "=" * 70)
print("IMAGE VISUAL STATISTICS")
print("=" * 70)

image_stats = image_statistics(
    train_df,
    TRAIN_DIR
)

print(
    f"Images analyzed: "
    f"{len(image_stats):,}"
)

print("\nBrightness statistics")

print(
    image_stats["brightness"]
    .describe()
)

print("\nContrast statistics")

print(
    image_stats["contrast"]
    .describe()
)


# ============================================================
# EXTREME BRIGHTNESS / DARKNESS
# ============================================================

brightness_low = image_stats[
    image_stats["brightness"] < 30
]

brightness_high = image_stats[
    image_stats["brightness"] > 220
]


print("\n" + "=" * 70)
print("EXTREME IMAGE BRIGHTNESS")
print("=" * 70)

print(
    f"Very dark images  : "
    f"{len(brightness_low)}"
)

print(
    f"Very bright images: "
    f"{len(brightness_high)}"
)


# ============================================================
# BRIGHTNESS DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(9, 5)
)

plt.hist(
    image_stats["brightness"],
    bins=40
)

plt.title(
    "Image Brightness Distribution"
)

plt.xlabel(
    "Mean Pixel Brightness"
)

plt.ylabel(
    "Frequency"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "brightness_distribution.png",
    dpi=200
)

plt.close()


# ============================================================
# CONTRAST DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(9, 5)
)

plt.hist(
    image_stats["contrast"],
    bins=40
)

plt.title(
    "Image Contrast Distribution"
)

plt.xlabel(
    "Mean Pixel Standard Deviation"
)

plt.ylabel(
    "Frequency"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "contrast_distribution.png",
    dpi=200
)

plt.close()


# ============================================================
# LOW-INFORMATION IMAGES
# ============================================================

# Very low contrast can indicate:
# - blank-ish images
# - heavily blurred images
# - unusual photographs
#
# This is NOT automatically label noise.
# We only flag them for inspection.

low_contrast = image_stats[
    image_stats["contrast"] < 20
].copy()


print("\n" + "=" * 70)
print("LOW-CONTRAST IMAGES")
print("=" * 70)

print(
    f"Low contrast images: "
    f"{len(low_contrast)}"
)

if len(low_contrast) > 0:

    print(
        low_contrast[
            [
                "filename",
                "brightness",
                "contrast",
                "label_combination"
            ]
        ]
        .head(20)
        .to_string(index=False)
    )


# ============================================================
# SAVE IMAGE STATISTICS
# ============================================================

image_stats.to_csv(
    OUTPUT_DIR
    / "image_statistics.csv",
    index=False
)


# ============================================================
# CROSS-SPLIT DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("TRAIN / VALIDATION / TEST CONSISTENCY")
print("=" * 70)

split_distribution = pd.DataFrame({

    "Train":
        train_df[CLASS_NAMES].mean(),

    "Validation":
        valid_df[CLASS_NAMES].mean(),

    "Test":
        test_df[CLASS_NAMES].mean()

}) * 100


print(
    split_distribution.round(2)
)


# ============================================================
# CROSS-SPLIT COMBINATION ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("COMBINATION DISTRIBUTION ACROSS SPLITS")
print("=" * 70)

combination_split = pd.DataFrame({

    "Train":
        train_df["label_combination"]
        .value_counts(),

    "Validation":
        valid_df["label_combination"]
        .value_counts(),

    "Test":
        test_df["label_combination"]
        .value_counts()

}).fillna(0)


print(
    combination_split
)


combination_split.to_csv(
    OUTPUT_DIR
    / "combination_split_distribution.csv"
)


# ============================================================
# FILE HASHING
# ============================================================

def md5_hash(path):

    hash_md5 = hashlib.md5()

    with open(path, "rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):

            hash_md5.update(chunk)

    return hash_md5.hexdigest()


def calculate_hashes(
    df,
    split_dir
):

    hashes = {}

    for filename in df["filename"]:

        path = split_dir / filename

        try:

            file_hash = md5_hash(path)

            hashes[filename] = file_hash

        except Exception:

            pass

    return hashes


# ============================================================
# CROSS-SPLIT DUPLICATE CONTENT
# ============================================================

print("\n" + "=" * 70)
print("CROSS-SPLIT IMAGE CONTENT DUPLICATE CHECK")
print("=" * 70)

print(
    "Calculating hashes..."
)

train_hashes = calculate_hashes(
    train_df,
    TRAIN_DIR
)

valid_hashes = calculate_hashes(
    valid_df,
    VALID_DIR
)

test_hashes = calculate_hashes(
    test_df,
    TEST_DIR
)


train_hash_set = set(
    train_hashes.values()
)

valid_hash_set = set(
    valid_hashes.values()
)

test_hash_set = set(
    test_hashes.values()
)


print(
    "Train ∩ Validation:",
    len(
        train_hash_set
        &
        valid_hash_set
    )
)

print(
    "Train ∩ Test:",
    len(
        train_hash_set
        &
        test_hash_set
    )
)

print(
    "Validation ∩ Test:",
    len(
        valid_hash_set
        &
        test_hash_set
    )
)


# ============================================================
# IDENTIFY POTENTIALLY SUSPICIOUS IMAGES
# ============================================================

# Important:
# These are NOT confirmed label errors.
# They are simply candidates for human inspection.

suspicious = image_stats[
    (
        image_stats["contrast"] < 20
    )
    |
    (
        image_stats["brightness"] < 30
    )
    |
    (
        image_stats["brightness"] > 220
    )
].copy()


print("\n" + "=" * 70)
print("POTENTIALLY SUSPICIOUS IMAGES")
print("=" * 70)

print(
    f"Candidates: "
    f"{len(suspicious)}"
)


suspicious.to_csv(
    OUTPUT_DIR
    / "suspicious_images.csv",
    index=False
)


# ============================================================
# SUMMARY REPORT
# ============================================================

multi_label_train = (
    train_df["num_labels"] > 1
).sum()

single_label_train = (
    train_df["num_labels"] == 1
).sum()

four_label_train = (
    train_df["num_labels"] == 4
).sum()


summary = {

    "train_images":
        len(train_df),

    "validation_images":
        len(valid_df),

    "test_images":
        len(test_df),

    "single_label_train":
        single_label_train,

    "multi_label_train":
        multi_label_train,

    "four_label_train":
        four_label_train,

    "multi_label_percentage":
        multi_label_train
        / len(train_df)
        * 100,

    "class_imbalance_ratio":
        imbalance_ratio,

    "very_dark_images_sample":
        len(brightness_low),

    "very_bright_images_sample":
        len(brightness_high),

    "low_contrast_images_sample":
        len(low_contrast),

    "suspicious_candidates_sample":
        len(suspicious)

}


summary_df = pd.DataFrame(
    list(summary.items()),
    columns=[
        "metric",
        "value"
    ]
)


summary_df.to_csv(
    OUTPUT_DIR
    / "quality_summary.csv",
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("QUALITY ANALYSIS COMPLETE")
print("=" * 70)

print(
    f"\nOutput directory:\n"
    f"{OUTPUT_DIR}"
)

print("\nGenerated files:")

for file in sorted(
    OUTPUT_DIR.iterdir()
):

    print(
        f"  - {file.name}"
    )

print("\n" + "=" * 70)