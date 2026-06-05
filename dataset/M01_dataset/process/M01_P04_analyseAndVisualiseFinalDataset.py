# ========= FILE NAME: M01_P04_analyseAndVisualiseFinalDataset.py =========
# FILE ROLE: Draws class distribution bar chart and writes dataset summary markdown
# TOTAL FUNCTIONS IN THIS FILE: 4
# FUNCTION INDEX RANGE: P04_F13 → P04_F16

# import OS utilities for file and directory handling
import os

# import matplotlib base module
import matplotlib

# force matplotlib to use a non-GUI backend (important for servers / headless environments)
# this allows saving plots directly to files without opening a window
matplotlib.use("Agg")

# import plotting interface
import matplotlib.pyplot as plt

# import paths, settings, and logging utilities
from dataset.M01_dataset.input.M01_I01_inputPaths import (
    M01_P03_OUTPUT_TRAIN_FOLDER,      # input: training dataset folder
    M01_P03_OUTPUT_VAL_FOLDER,        # input: validation dataset folder
    M01_P03_OUTPUT_TEST_FOLDER,       # input: test dataset folder

    M01_P04_OUTPUT_DISTRIBUTION_PNG,  # output path for bar chart image
    M01_P04_OUTPUT_SUMMARY_MD,        # output path for markdown summary
    M01_OUTPUT_REPORTS_FOLDER,        # reports folder (not directly used here but available)

    M01_P03_TRAIN_COUNT,              # expected train images per species
    M01_P03_VAL_COUNT,                # expected validation images per species
    M01_P03_TEST_COUNT,               # expected test images per species

    LOGGER,                           # logging function
    WARN_LOGGER                       # warning logging function
)


# FUNCTION 13
def P04_F13_countImagesInEachSplitAndSpecies(
    train_folder: str, val_folder: str, test_folder: str
) -> dict:
    # returns {species: {train: N, val: N, test: N}} for every species

    counts = {}  # dictionary to store results

    # map split names to their folders
    folder_map = {
        "train": train_folder,
        "val": val_folder,
        "test": test_folder
    }

    # iterate through each split
    for split_name, folder in folder_map.items():

        # iterate through species folders
        for species in sorted(os.listdir(folder)):

            path = os.path.join(folder, species)

            # skip non-directory items
            if not os.path.isdir(path):
                continue

            # count image files in this species folder
            n = len([f for f in os.listdir(path)
                     if f.lower().endswith((".png", ".jpg", ".jpeg"))])

            # initialize species entry if not already present
            if species not in counts:
                counts[species] = {}

            # store count for this split
            counts[species][split_name] = n

    return counts


# FUNCTION 14
def P04_F14_printSplitCountsTableToConsole(counts: dict) -> None:
    # prints a readable table of split counts so you can visually verify correctness

    print("\n--- FINAL SPLIT COUNTS ---")

    # print header row with alignment formatting
    print(f"  {'Species':<35} {'Train':>6} {'Val':>6} {'Test':>6} {'Total':>7}")

    # print separator line
    print(f"  {'-'*35} {'-'*6} {'-'*6} {'-'*6} {'-'*7}")

    # print one row per species
    for species, splits in sorted(counts.items()):

        tr = splits.get("train", 0)
        vl = splits.get("val",   0)
        te = splits.get("test",  0)

        print(f"  {species:<35} {tr:>6} {vl:>6} {te:>6} {tr+vl+te:>7}")

    # compute totals across all species
    total_train = sum(s.get("train", 0) for s in counts.values())
    total_val   = sum(s.get("val",   0) for s in counts.values())
    total_test  = sum(s.get("test",  0) for s in counts.values())

    # print totals row
    print(f"\n  {'TOTAL':<35} {total_train:>6} {total_val:>6} {total_test:>6} {total_train+total_val+total_test:>7}")
    print()


# FUNCTION 15
def P04_F15_drawBarChartOfClassDistributionAndSaveIt(
    counts: dict, output_png: str
) -> None:
    # draws a horizontal bar chart of train/val/test counts per species

    # ensure output directory exists
    os.makedirs(os.path.dirname(output_png), exist_ok=True)

    # sort species names for consistent ordering
    species_names = sorted(counts.keys())

    # extract counts for each split
    train_vals = [counts[s].get("train", 0) for s in species_names]
    val_vals   = [counts[s].get("val",   0) for s in species_names]
    test_vals  = [counts[s].get("test",  0) for s in species_names]

    # create y-axis positions
    y = range(len(species_names))

    # define bar height
    bar_h = 0.25

    # create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # draw horizontal bars for each split (slightly offset vertically)
    ax.barh([i + bar_h for i in y], train_vals, height=bar_h, label="train", color="#4C72B0")
    ax.barh([i         for i in y], val_vals,   height=bar_h, label="val",   color="#55A868")
    ax.barh([i - bar_h for i in y], test_vals,  height=bar_h, label="test",  color="#C44E52")

    # set y-axis ticks and labels (replace underscores with spaces for readability)
    ax.set_yticks(list(y))
    ax.set_yticklabels([s.replace("_", " ") for s in species_names], fontsize=9)

    # label x-axis
    ax.set_xlabel("Number of images")

    # set chart title
    ax.set_title("M01_P04 - Dataset split distribution (10 species)")

    # show legend
    ax.legend()

    # adjust layout to avoid clipping
    plt.tight_layout()

    # save figure to file
    plt.savefig(output_png, dpi=150)

    # close figure to free memory
    plt.close()

    # log output
    LOGGER("P04", f"bar chart saved → {output_png}")


# FUNCTION 16
def P04_F16_writeDatasetSummaryMarkdownFile(counts: dict, output_md: str) -> None:
    # writes a markdown summary file describing the dataset

    # ensure output directory exists
    os.makedirs(os.path.dirname(output_md), exist_ok=True)

    # compute dataset statistics
    total_species = len(counts)
    total_images  = sum(sum(s.values()) for s in counts.values())
    total_train   = sum(s.get("train", 0) for s in counts.values())
    total_val     = sum(s.get("val",   0) for s in counts.values())
    total_test    = sum(s.get("test",  0) for s in counts.values())

    # build markdown content line by line
    lines = [
        "# Path 4 — Dataset Summary",
        "",
        f"- **Total species:** {total_species}",
        f"- **Total images:** {total_images}",
        f"- **Train:** {total_train} ({M01_P03_TRAIN_COUNT} per species)",
        f"- **Validation:** {total_val} ({M01_P03_VAL_COUNT} per species)",
        f"- **Test:** {total_test} ({M01_P03_TEST_COUNT} per species)",
        f"- **Random seed:** 20",
        "",
        "## Per-species counts",
        "",
        "| Species | Train | Val | Test |",
        "|---------|-------|-----|------|",
    ]

    # add one table row per species
    for species, splits in sorted(counts.items()):
        tr = splits.get("train", 0)
        vl = splits.get("val",   0)
        te = splits.get("test",  0)

        lines.append(f"| {species.replace('_', ' ')} | {tr} | {vl} | {te} |")

    # write all lines to markdown file
    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # log completion
    LOGGER("P04", f"summary markdown saved → {output_md}")


# MAIN
if __name__ == "__main__":

    LOGGER("P04", "=== M01_P04 — analysing and visualising final dataset ===")

    # STEP 1 — count images
    LOGGER("P04", "step 1 — counting images in all splits...")

    counts = P04_F13_countImagesInEachSplitAndSpecies(
        M01_P03_OUTPUT_TRAIN_FOLDER,
        M01_P03_OUTPUT_VAL_FOLDER,
        M01_P03_OUTPUT_TEST_FOLDER
    )

    # STEP 2 — print table
    LOGGER("P04", "step 2 — printing counts table...")

    P04_F14_printSplitCountsTableToConsole(counts)

    # STEP 3 — draw chart
    LOGGER("P04", "step 3 — drawing bar chart...")

    P04_F15_drawBarChartOfClassDistributionAndSaveIt(
        counts,
        M01_P04_OUTPUT_DISTRIBUTION_PNG
    )

    # STEP 4 — write markdown
    LOGGER("P04", "step 4 — writing markdown summary...")

    P04_F16_writeDatasetSummaryMarkdownFile(
        counts,
        M01_P04_OUTPUT_SUMMARY_MD
    )

    LOGGER("P04", "=== M01_P04 done — check M01_dataset/output/reports/ ===")