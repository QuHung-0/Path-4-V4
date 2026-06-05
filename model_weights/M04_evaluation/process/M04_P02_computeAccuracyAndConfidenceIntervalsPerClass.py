# ========= FILE NAME: M04_P02_computeAccuracyAndConfidenceIntervalsPerClass.py =========
# FILE ROLE: Computes Wilson CI for each class and overall, plus F1/precision/recall per class
# TOTAL FUNCTIONS IN THIS FILE: 4
# FUNCTION INDEX RANGE: P02_F04 → P02_F07

# import OS module for directory creation
import os

# import CSV module for reading and writing tables
import csv

# import math for square roots and confidence-interval math
import math

# import defaultdict so per-class counts can be initialized automatically
from collections import defaultdict

# import paths, settings, and logger
from model_weights.M04_evaluation.input.M04_I01_inputPaths import (
    M04_P01_OUTPUT_PREDICTIONS_CSV,     # input CSV from M04_P01

    M04_P02_OUTPUT_PER_CLASS_CI_CSV,    # output CSV for Wilson intervals
    M04_P02_OUTPUT_F1_CSV,              # output CSV for precision/recall/F1

    M04_CI_CONFIDENCE_LEVEL,            # confidence level used for Wilson intervals

    LOGGER                              # logger
)


# FUNCTION 4
def P02_F04_computeOverallAccuracyFromPredictionsCSV(predictions_csv: str) -> dict:
    # reads the predictions CSV and returns total correct, total images, and overall accuracy
    rows = []

    # load all rows from the CSV file
    with open(predictions_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # total number of predictions
    total = len(rows)

    # count how many predictions are correct
    correct = sum(int(r["correct"]) for r in rows)

    # compute accuracy percentage
    acc = 100.0 * correct / total

    LOGGER("P02", f"overall: {correct}/{total} correct = {acc:.2f}%")

    # return summary plus raw rows for downstream functions
    return {"total": total, "correct": correct, "accuracy_pct": acc, "rows": rows}


# FUNCTION 5
def P02_F05_computeWilsonConfidenceIntervalForOneClass(
    correct: int, total: int, confidence_level: float = 0.95
) -> tuple:
    # Wilson score interval — more accurate than normal approximation for small samples
    # returns (lower_bound_pct, upper_bound_pct)

    # if there are no samples, return a zero-width interval
    if total == 0:
        return 0.0, 0.0

    # z-score for the requested confidence level
    # 1.96 is used for 95%, 2.576 is used for about 99%
    z = 1.96 if confidence_level == 0.95 else 2.576

    # observed proportion
    p = correct / total

    # sample size
    n = total

    # Wilson centre term
    centre = (p + z*z/(2*n)) / (1 + z*z/n)

    # Wilson margin term
    margin = (z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / (1 + z*z/n)

    # convert to percentage and clamp to [0, 100]
    lower = max(0.0, (centre - margin) * 100)
    upper = min(100.0, (centre + margin) * 100)

    return round(lower, 2), round(upper, 2)


# FUNCTION 6
def P02_F06_computeWilsonIntervalForAllClassesAndSave(rows: list, output_csv: str) -> None:
    # groups rows by true label, computes per-class accuracy and Wilson CI, saves to CSV

    # store per-class counts
    per_class = defaultdict(lambda: {"correct": 0, "total": 0})

    # accumulate correct/total counts for each true label
    for r in rows:
        label = r["true_label"]
        per_class[label]["total"] += 1
        per_class[label]["correct"] += int(r["correct"])

    # ensure output folder exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # list of result rows to write to CSV
    results = []

    print("\n--- PER-CLASS ACCURACY WITH 95% WILSON CI ---")
    print(f"  {'Species':<35} {'Correct':>8} {'Total':>6} {'Acc%':>7} {'CI Lower':>9} {'CI Upper':>9}")
    print(f"  {'-'*35} {'-'*8} {'-'*6} {'-'*7} {'-'*9} {'-'*9}")

    # compute accuracy and CI for each species
    for species in sorted(per_class.keys()):
        c = per_class[species]["correct"]
        t = per_class[species]["total"]
        acc = 100.0 * c / t
        lo, hi = P02_F05_computeWilsonConfidenceIntervalForOneClass(c, t, M04_CI_CONFIDENCE_LEVEL)

        print(f"  {species:<35} {c:>8} {t:>6} {acc:>6.2f}% [{lo:>7.2f}%, {hi:>7.2f}%]")

        results.append({
            "species": species,
            "correct": c,
            "total": t,
            "accuracy_pct": round(acc, 2),
            "ci_lower_pct": lo,
            "ci_upper_pct": hi,
        })

    # compute overall totals across all classes
    total_c = sum(per_class[s]["correct"] for s in per_class)
    total_t = sum(per_class[s]["total"] for s in per_class)

    # compute overall Wilson interval
    lo_all, hi_all = P02_F05_computeWilsonConfidenceIntervalForOneClass(total_c, total_t)

    print(f"\n  {'OVERALL':<35} {total_c:>8} {total_t:>6} {100*total_c/total_t:>6.2f}% [{lo_all:>7.2f}%, {hi_all:>7.2f}%]")
    print()

    # append overall summary row
    results.append({
        "species": "OVERALL",
        "correct": total_c,
        "total": total_t,
        "accuracy_pct": round(100.0 * total_c / total_t, 2),
        "ci_lower_pct": lo_all,
        "ci_upper_pct": hi_all,
    })

    # save results to CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["species", "correct", "total", "accuracy_pct", "ci_lower_pct", "ci_upper_pct"]
        )
        writer.writeheader()
        writer.writerows(results)

    LOGGER("P02", f"per-class CI saved → {output_csv}")


# FUNCTION 7
def P02_F07_computePrecisionRecallF1ForEachClass(rows: list, output_csv: str) -> None:
    # computes TP, FP, FN per class then derives precision, recall, F1

    # identify all classes present in the dataset
    classes = sorted(set(r["true_label"] for r in rows))

    # initialize counters
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    # accumulate confusion counts
    for r in rows:
        true = r["true_label"]
        pred = r["predicted_label"]
        if true == pred:
            tp[true] += 1
        else:
            fn[true] += 1
            fp[pred] += 1

    # ensure output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    print("--- PER-CLASS PRECISION / RECALL / F1 ---")
    print(f"  {'Species':<35} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print(f"  {'-'*35} {'-'*10} {'-'*8} {'-'*8}")

    # list of rows to write to CSV
    results = []

    # compute metrics per class
    for cls in classes:
        precision = tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) > 0 else 0.0
        recall = tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f"  {cls:<35} {precision*100:>9.2f}% {recall*100:>7.2f}% {f1*100:>7.2f}%")

        results.append({
            "species": cls,
            "precision_pct": round(precision * 100, 2),
            "recall_pct": round(recall * 100, 2),
            "f1_pct": round(f1 * 100, 2),
        })

    print()

    # save per-class PR/F1 table to CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["species", "precision_pct", "recall_pct", "f1_pct"])
        writer.writeheader()
        writer.writerows(results)

    LOGGER("P02", f"F1/precision/recall saved → {output_csv}")


# MAIN
if __name__ == "__main__":
    # entry point for the evaluation metrics script

    LOGGER("P02", "=== M04_P02 — computing accuracy and confidence intervals ===")

    LOGGER("P02", "step 1 — loading predictions CSV...")

    # load the prediction rows produced by M04_P01
    result = P02_F04_computeOverallAccuracyFromPredictionsCSV(M04_P01_OUTPUT_PREDICTIONS_CSV)
    rows = result["rows"]

    LOGGER("P02", "step 2 — computing Wilson CI per class...")

    # compute per-class Wilson intervals and save to CSV
    P02_F06_computeWilsonIntervalForAllClassesAndSave(rows, M04_P02_OUTPUT_PER_CLASS_CI_CSV)

    LOGGER("P02", "step 3 — computing precision / recall / F1 per class...")

    # compute per-class F1-related metrics and save to CSV
    P02_F07_computePrecisionRecallF1ForEachClass(rows, M04_P02_OUTPUT_F1_CSV)

    LOGGER("P02", "=== M04_P02 done ===")