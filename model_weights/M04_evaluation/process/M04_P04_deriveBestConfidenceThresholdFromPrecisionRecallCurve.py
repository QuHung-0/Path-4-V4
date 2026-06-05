# ========= FILE NAME: M04_P04_deriveBestConfidenceThresholdFromPrecisionRecallCurve.py =========
# FILE ROLE: Plots precision vs recall at every threshold and saves the threshold that maximises F1
# TOTAL FUNCTIONS IN THIS FILE: 3
# FUNCTION INDEX RANGE: P04_F11 → P04_F13

# import OS for folder creation
import os

# import CSV for reading predictions
import csv

# import JSON for saving the chosen threshold
import json

# set matplotlib backend to file-only mode
import matplotlib
matplotlib.use("Agg")

# import plotting tools
import matplotlib.pyplot as plt

# import paths and logger
from model_weights.M04_evaluation.input.M04_I01_inputPaths import (
    M04_P01_OUTPUT_PREDICTIONS_CSV,   # input predictions CSV

    M04_P04_OUTPUT_PR_CURVE_PNG,      # output PR curve figure
    M04_P04_OUTPUT_THRESHOLD_JSON,    # output best threshold JSON

    LOGGER                           # logger
)


# FUNCTION 11
def P04_F11_computePrecisionAndRecallAtEveryConfidenceThreshold(
    predictions_csv: str,
) -> list:
    # steps through thresholds from 0.50 to 0.99 in 0.01 steps
    # at each threshold, counts predictions that meet it and computes precision and recall

    rows = []

    # load prediction rows
    with open(predictions_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # total number of images in the test set
    total_positives = len(rows)

    # list of threshold curve points
    curve = []

    # test thresholds from 0.50 to 0.99 inclusive of 0.50, exclusive of 1.00
    for t_int in range(50, 100):
        threshold = t_int / 100.0

        # keep only predictions with confidence at or above the threshold
        accepted = [r for r in rows if float(r["confidence"]) >= threshold]

        # stop if nothing survives this threshold
        if len(accepted) == 0:
            break

        # count correct accepted predictions
        tp = sum(int(r["correct"]) for r in accepted)

        # precision among accepted predictions
        precision = tp / len(accepted)

        # recall across all test images
        recall = tp / total_positives

        # F1 score at this threshold
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        # coverage = percentage of test set accepted by this threshold
        coverage = 100.0 * len(accepted) / total_positives

        # store one threshold point
        curve.append({
            "threshold": threshold,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "coverage_pct": round(coverage, 2),
            "accepted": len(accepted),
        })

    LOGGER("P04", f"computed {len(curve)} threshold points from 0.50 to 0.99")
    return curve


# FUNCTION 12
def P04_F12_findThresholdThatMaximisesF1Score(curve: list) -> dict:
    # finds the point in the curve with the highest F1 score

    # select the curve point with maximum F1
    best = max(curve, key=lambda x: x["f1"])

    # print the chosen threshold and its metrics
    LOGGER("P04", f"best threshold by F1:  {best['threshold']:.2f}")
    LOGGER("P04", f"  precision: {best['precision']*100:.2f}%")
    LOGGER("P04", f"  recall:    {best['recall']*100:.2f}%")
    LOGGER("P04", f"  F1:        {best['f1']*100:.2f}%")
    LOGGER("P04", f"  coverage:  {best['coverage_pct']:.2f}% of test images accepted")

    # print a table of the whole threshold sweep
    print("\n--- THRESHOLD SUMMARY ---")
    print(f"  {'Threshold':>10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Coverage':>10}")
    print(f"  {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*10}")

    for pt in curve:
        marker = "  ← BEST" if pt["threshold"] == best["threshold"] else ""
        print(
            f"  {pt['threshold']:>10.2f} {pt['precision']*100:>9.2f}% "
            f"{pt['recall']*100:>7.2f}% {pt['f1']*100:>7.2f}% "
            f"{pt['coverage_pct']:>9.2f}%{marker}"
        )
    print()

    return best


# FUNCTION 13
def P04_F13_savePrecisionRecallCurveAndBestThresholdToFile(
    curve: list, best: dict, output_png: str, output_json: str
) -> None:
    # saves the PR curve as a PNG and the best threshold to a JSON file

    # ensure output directory exists
    os.makedirs(os.path.dirname(output_png), exist_ok=True)

    # extract data series from the threshold curve
    thresholds = [pt["threshold"] for pt in curve]
    precisions = [pt["precision"] for pt in curve]
    recalls = [pt["recall"] for pt in curve]
    f1s = [pt["f1"] for pt in curve]

    # create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # plot precision, recall, and F1 against threshold
    ax.plot(thresholds, precisions, label="Precision", color="#4C72B0", linewidth=2)
    ax.plot(thresholds, recalls, label="Recall", color="#55A868", linewidth=2)
    ax.plot(thresholds, f1s, label="F1", color="#C44E52", linewidth=2)

    # mark the best threshold with a vertical line
    ax.axvline(
        x=best["threshold"],
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label=f"Best threshold = {best['threshold']:.2f}"
    )

    # style axes and title
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Score")
    # ax.set_title("Path 4 — Precision / Recall / F1 vs Confidence Threshold")
    ax.set_title("M04_P04 - Precision / Recall / F1 vs Confidence Threshold") #No path appear in the title
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(0.50, 1.00)
    ax.set_ylim(0.0, 1.05)

    # save and close figure
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()
    LOGGER("P04", f"PR curve PNG saved → {output_png}")

    # save best threshold information as JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({
            "threshold": best["threshold"],
            "precision_pct": round(best["precision"] * 100, 2),
            "recall_pct": round(best["recall"] * 100, 2),
            "f1_pct": round(best["f1"] * 100, 2),
            "coverage_pct": best["coverage_pct"],
        }, f, indent=4)

    LOGGER("P04", f"best threshold JSON saved → {output_json}")


# MAIN
if __name__ == "__main__":
    # direct-run entry point

    LOGGER("P04", "=== M04_P04 — deriving best confidence threshold from PR curve ===")

    LOGGER("P04", "step 1 — computing precision/recall at every threshold...")

    # build threshold curve from predictions
    curve = P04_F11_computePrecisionAndRecallAtEveryConfidenceThreshold(M04_P01_OUTPUT_PREDICTIONS_CSV)

    LOGGER("P04", "step 2 — finding threshold that maximises F1...")

    # choose the best threshold by F1
    best = P04_F12_findThresholdThatMaximisesF1Score(curve)

    LOGGER("P04", "step 3 — saving PR curve PNG and threshold JSON...")

    # save plot and JSON summary
    P04_F13_savePrecisionRecallCurveAndBestThresholdToFile(
        curve, best, M04_P04_OUTPUT_PR_CURVE_PNG, M04_P04_OUTPUT_THRESHOLD_JSON
    )

    LOGGER("P04", "=== M04_P04 done ===")
    LOGGER("P04", "from this point on, all modules read the threshold from:")
    LOGGER("P04", f"  {M04_P04_OUTPUT_THRESHOLD_JSON}")
    LOGGER("P04", f"  (never hardcode a confidence threshold number again)")