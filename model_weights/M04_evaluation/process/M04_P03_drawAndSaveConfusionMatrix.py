# ========= FILE NAME: M04_P03_drawAndSaveConfusionMatrix.py =========
# FILE ROLE: Builds confusion matrix from predictions and saves it as a heatmap PNG
# TOTAL FUNCTIONS IN THIS FILE: 3
# FUNCTION INDEX RANGE: P03_F08 → P03_F10

# import OS utilities for output folder creation
import os

# import CSV reader for loading prediction results
import csv

# import defaultdict for flexible matrix construction
from collections import defaultdict

# set matplotlib backend to file-only mode
import matplotlib
matplotlib.use("Agg")

# import plotting and heatmap utilities
import matplotlib.pyplot as plt
import seaborn as sns

# import paths and logger
from model_weights.M04_evaluation.input.M04_I01_inputPaths import (
    M04_P01_OUTPUT_PREDICTIONS_CSV,   # predictions CSV from M04_P01

    M04_P03_OUTPUT_CONFUSION_PNG,     # confusion matrix image output

    LOGGER                            # logger
)


# FUNCTION 8
def P03_F08_buildConfusionMatrixFromPredictionsAndTrueLabels(predictions_csv: str) -> tuple:
    # reads predictions CSV, builds a 2D count matrix [true][predicted]
    # returns (matrix_dict, sorted_class_list)

    rows = []

    # load all prediction rows
    with open(predictions_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # collect all class names appearing in true labels
    classes = sorted(set(r["true_label"] for r in rows))

    # matrix[true][predicted] = count
    matrix = defaultdict(lambda: defaultdict(int))

    # increment the appropriate cell for each prediction
    for r in rows:
        matrix[r["true_label"]][r["predicted_label"]] += 1

    LOGGER("P03", f"confusion matrix built — {len(classes)} classes, {len(rows)} predictions")
    return matrix, classes


# FUNCTION 9
def P03_F09_drawHeatmapOfConfusionMatrixAndSaveAsPng(
    matrix: dict, classes: list, output_png: str
) -> None:
    # converts matrix dict to a 2D list and draws a seaborn annotated heatmap

    # number of classes
    n = len(classes)

    # convert sparse dictionary matrix into dense 2D list in class order
    data = [[matrix[true][pred] for pred in classes] for true in classes]

    # make long class names easier to read by wrapping underscores onto new lines
    short = [c.replace("_", "\n") for c in classes]

    # create figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))

    # draw heatmap with cell counts
    sns.heatmap(
        data,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=short,
        yticklabels=short,
        linewidths=0.5,
        ax=ax,
    )

    # label axes and title
    ax.set_xlabel("Predicted species", fontsize=11, labelpad=10)
    ax.set_ylabel("True species", fontsize=11, labelpad=10)
    ax.set_title("M04_P03 - Confusion Matrix (test set, 300 images)", fontsize=13, pad=14)

    # style tick labels
    plt.xticks(fontsize=8, rotation=45, ha="right")
    plt.yticks(fontsize=8, rotation=0)

    # make layout fit nicely
    plt.tight_layout()

    # ensure output folder exists
    os.makedirs(os.path.dirname(output_png), exist_ok=True)

    # save image and close figure
    plt.savefig(output_png, dpi=150)
    plt.close()

    LOGGER("P03", f"confusion matrix PNG saved → {output_png}")


# FUNCTION 10
def P03_F10_identifyWhichClassesGetMostOftenConfusedWithEachOther(
    matrix: dict, classes: list
) -> None:
    # prints the top off-diagonal confusions so you can see which pairs are hardest

    # collect all off-diagonal mistakes as tuples: (count, true_class, predicted_class)
    confusions = []
    for true in classes:
        for pred in classes:
            if true != pred and matrix[true][pred] > 0:
                confusions.append((matrix[true][pred], true, pred))

    # sort by most frequent mistake first
    confusions.sort(reverse=True)

    print("\n--- TOP CONFUSIONS (off-diagonal) ---")
    if not confusions:
        print("  none — perfect classification!")
    else:
        for count, true, pred in confusions[:10]:
            print(f"  {true}  →  {pred}   ({count} times)")
    print()


# MAIN
if __name__ == "__main__":
    # direct-run entry point

    LOGGER("P03", "=== M04_P03 — drawing confusion matrix ===")

    LOGGER("P03", "step 1 — building matrix from predictions CSV...")

    # build matrix from saved predictions
    matrix, classes = P03_F08_buildConfusionMatrixFromPredictionsAndTrueLabels(M04_P01_OUTPUT_PREDICTIONS_CSV)

    LOGGER("P03", "step 2 — drawing heatmap...")

    # draw and save confusion matrix figure
    P03_F09_drawHeatmapOfConfusionMatrixAndSaveAsPng(matrix, classes, M04_P03_OUTPUT_CONFUSION_PNG)

    LOGGER("P03", "step 3 — identifying worst confusions...")

    # print the most common mistakes
    P03_F10_identifyWhichClassesGetMostOftenConfusedWithEachOther(matrix, classes)

    LOGGER("P03", "=== M04_P03 done ===")