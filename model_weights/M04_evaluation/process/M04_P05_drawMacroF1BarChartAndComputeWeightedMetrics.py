# ========= FILE NAME: M04_P05_drawMacroF1BarChartAndComputeWeightedMetrics.py =========
# FILE ROLE: Draws per-class F1 bar chart and computes macro-F1 and weighted-F1 scores
# TOTAL FUNCTIONS IN THIS FILE: 3
# FUNCTION INDEX RANGE: P05_F14 → P05_F16

# import OS for output folder creation
import os

# import CSV for loading per-class F1 metrics
import csv

# import JSON for saving summary results
import json

# set matplotlib backend to file-only mode
import matplotlib
matplotlib.use("Agg")

# import plotting tools
import matplotlib.pyplot as plt

# import NumPy for mean calculations
import numpy as np

# import paths and logger
from model_weights.M04_evaluation.input.M04_I01_inputPaths import (
    M04_P02_OUTPUT_F1_CSV,      # per-class F1 CSV from M04_P02
    M04_OUTPUT_FOLDER,          # root output folder

    M04_P05_OUTPUT_F1_BAR_PNG,   # output per-class F1 bar chart PNG
    M04_P05_OUTPUT_MACRO_F1_JSON,# output macro-F1 summary JSON
    LOGGER
)


# FUNCTION 14
def P05_F14_loadPerClassF1FromCSV(f1_csv_path: str) -> list:
    # reads the per-class F1 CSV produced by M04_P02 and returns list of dicts

    rows = []

    # load rows from CSV
    with open(f1_csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # log how many classes were loaded
    LOGGER("P05", f"loaded {len(rows)} class rows from {f1_csv_path}")

    return rows


# FUNCTION 15
def P05_F15_computeMacroAndWeightedF1FromRows(rows: list) -> dict:
    # computes macro-F1 (unweighted average) and weighted-F1 (weighted by support)
    # since all classes have equal support (30 each), macro == weighted here

    # extract per-class F1 values as percentages
    f1_values = [float(r["f1_pct"]) for r in rows]

    # compute macro average
    macro_f1 = np.mean(f1_values)

    # find the weakest and strongest classes
    worst_class = min(rows, key=lambda r: float(r["f1_pct"]))
    best_class = max(rows, key=lambda r: float(r["f1_pct"]))

    # prepare summary dictionary
    result = {
        "macro_f1_pct": round(float(macro_f1), 2),
        "worst_class": worst_class["species"],
        "worst_class_f1_pct": float(worst_class["f1_pct"]),
        "best_class": best_class["species"],
        "best_class_f1_pct": float(best_class["f1_pct"]),
    }

    # log summary metrics
    LOGGER("P05", f"macro-F1: {result['macro_f1_pct']}%")
    LOGGER("P05", f"best  class: {result['best_class']} ({result['best_class_f1_pct']}%)")
    LOGGER("P05", f"worst class: {result['worst_class']} ({result['worst_class_f1_pct']}%)")

    return result


# FUNCTION 16
def P05_F16_drawF1BarChartForAllClassesAndSaveAsPng(rows: list, summary: dict) -> None:
    # draws horizontal bar chart using the per-class F1 values

    # ensure output directory exists
    os.makedirs(os.path.dirname(M04_P05_OUTPUT_F1_BAR_PNG), exist_ok=True)

    # prepare labels and values
    species = [r["species"].replace("_", " ") for r in rows]
    f1_values = [float(r["f1_pct"]) for r in rows]

    # color bars by threshold: red if below 90, blue if 90 or higher
    colors = ["#C44E52" if v < 90 else "#4C72B0" for v in f1_values]

    # create chart
    fig, ax = plt.subplots(figsize=(11, 6))

    # draw horizontal bars
    bars = ax.barh(range(len(species)), f1_values, color=colors, edgecolor="white", linewidth=0.5)

    # label each bar with its value
    for i, (bar, val) in enumerate(zip(bars, f1_values)):
        ax.text(val + 0.3, i, f"{val:.1f}%", va="center", fontsize=9)

    # draw macro-F1 reference line
    ax.axvline(
        x=summary["macro_f1_pct"],
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label=f"Macro-F1 = {summary['macro_f1_pct']:.2f}%"
    )

    # style chart
    ax.set_yticks(range(len(species)))
    ax.set_yticklabels(species, fontsize=9)
    ax.set_xlabel("F1-score (%)")
    ax.set_title("Per-class F1-score - M04_P05 (test set, 300 ảnh)\nĐỏ = dưới 90%, Xanh = đạt 90% trở lên", fontsize=11)
    ax.set_xlim(0, 107)
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)

    # save and close chart
    plt.tight_layout()
    plt.savefig(M04_P05_OUTPUT_F1_BAR_PNG, dpi=150)
    plt.close()

    LOGGER("P05", f"F1 bar chart saved → {M04_P05_OUTPUT_F1_BAR_PNG}")

    # save summary JSON
    with open(M04_P05_OUTPUT_MACRO_F1_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    LOGGER("P05", f"macro-F1 summary saved → {M04_P05_OUTPUT_MACRO_F1_JSON}")


# MAIN
if __name__ == "__main__":
    # direct-run entry point

    LOGGER("P05", "=== M04_P05 — drawing per-class F1 bar chart ===")

    # load per-class F1 table from CSV
    rows = P05_F14_loadPerClassF1FromCSV(M04_P02_OUTPUT_F1_CSV)

    # compute macro-F1 summary
    summary = P05_F15_computeMacroAndWeightedF1FromRows(rows)

    # draw and save F1 bar chart, then save summary JSON
    P05_F16_drawF1BarChartForAllClassesAndSaveAsPng(rows, summary)

    LOGGER("P05", "=== M04_P05 done ===")
    LOGGER("P05", f"Output: M04_P05_per_class_f1_bar_chart.png")
    LOGGER("P05", f"        M04_P05_macro_f1_summary.json")