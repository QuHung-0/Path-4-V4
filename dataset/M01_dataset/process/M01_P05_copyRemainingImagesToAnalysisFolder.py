# ========= FILE NAME: M01_P05_copyRemainingImagesToAnalysisFolder.py =========
# FILE ROLE: Copies every image from Fish4Knowledge that was NOT selected into
#            the 2000-image balanced dataset, into a single analysis folder.
#
#            Two groups are created:
#            1. known_species/  — remaining images from the 10 trained species
#               (each species had 200 taken; this copies what was left behind)
#            2. excluded_species/ — ALL images from the 13 species that had
#               fewer than 200 images and were never trained on at all
#
#            M07_P06 then runs bulk inference on this folder to answer:
#            "how does the model behave on Fish4Knowledge images it has never seen?"
#
# TOTAL FUNCTIONS IN THIS FILE: 4
# FUNCTION INDEX RANGE: P05_F17 → P05_F20
# RUN ORDER: after M01_P04

import os
import json
import shutil
from datetime import datetime

# ── path constants ─────────────────────────────────────────────────────────────
# M01_I01 only re-exports constants that existed at V4 creation time.
# M01_P05_OUTPUT_REMAINING_FOLDER and M01_P05_OUTPUT_REMAINING_REPORT_TXT are
# new additions to M00_C01, so we import them directly from M00_C01 here.
# All other M01 constants come from M01_I01 as usual.
from dataset.M01_dataset.input.M01_I01_inputPaths import (
    M01_INPUT_RAW_FISH_IMAGE_FOLDER,     # path to raw fish_image folder with fish_01 ... fish_23
    M01_P02_INPUT_SPECIES_MAP,           # JSON mapping fish_XX -> species name
    M01_P02_OUTPUT_BALANCED_FOLDER,      # folder containing the 200-image balanced set per species
    M01_OUTPUT_REPORTS_FOLDER,           # shared reports folder

    M01_P05_OUTPUT_REMAINING_FOLDER,  # output root — known_species/ and excluded_species/ live here
    M01_P05_OUTPUT_REMAINING_REPORT_TXT,  # text report summarising what was copied

    LOGGER,
    WARN_LOGGER,
)

# accepted image extensions — same set as M01_P01
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


# FUNCTION 17
def P05_F17_loadSpeciesMapAndIdentifyKnownAndExcludedFolders() -> tuple:
    # loads the species map JSON and splits the 23 raw folders into:
    #   known_map  — dict {fish_XX: species_name} for the 10 trained species
    #   excluded   — list of fish_XX folder names for the 13 excluded species

    with open(M01_P02_INPUT_SPECIES_MAP, "r", encoding="utf-8") as f:
        species_map = json.load(f)   # {fish_XX: species_name}

    # collect every folder that actually exists in the raw dataset
    all_raw_folders = sorted([
        d for d in os.listdir(M01_INPUT_RAW_FISH_IMAGE_FOLDER)
        if os.path.isdir(os.path.join(M01_INPUT_RAW_FISH_IMAGE_FOLDER, d))
    ])

    known_map = {}   # fish_XX -> species_name for the 10 trained species
    excluded  = []   # fish_XX for the 13 species never trained on

    for folder in all_raw_folders:
        if folder in species_map:
            known_map[folder] = species_map[folder]
        else:
            excluded.append(folder)

    LOGGER("P05", f"known species folders   : {len(known_map)}")
    LOGGER("P05", f"excluded species folders: {len(excluded)}")
    return known_map, excluded


# FUNCTION 18
def P05_F18_copyRemainingImagesForKnownSpecies(known_map: dict) -> dict:
    # for each trained species, finds which images were NOT selected into the
    # 200-image balanced set and copies them to known_species/<species_name>/
    # returns a dict {species_name: count_copied}

    summary        = {}
    known_out_root = os.path.join(M01_P05_OUTPUT_REMAINING_FOLDER, "known_species")

    for fish_folder, species_name in known_map.items():
        raw_dir      = os.path.join(M01_INPUT_RAW_FISH_IMAGE_FOLDER, fish_folder)
        balanced_dir = os.path.join(M01_P02_OUTPUT_BALANCED_FOLDER, species_name)
        out_dir      = os.path.join(known_out_root, species_name)
        os.makedirs(out_dir, exist_ok=True)

        # all image filenames in the raw folder for this species
        raw_files = {
            f for f in os.listdir(raw_dir)
            if os.path.splitext(f)[1].lower() in _IMAGE_EXTS
        }

        # the 200 filenames already selected into the balanced set
        if os.path.isdir(balanced_dir):
            selected_files = {
                f for f in os.listdir(balanced_dir)
                if os.path.splitext(f)[1].lower() in _IMAGE_EXTS
            }
        else:
            # balanced folder missing — warn and treat all raw as remaining
            selected_files = set()
            WARN_LOGGER("P05", f"balanced folder not found for {species_name} — treating all as remaining")

        # remaining = raw minus selected
        remaining_files = sorted(raw_files - selected_files)

        copied = 0
        for fname in remaining_files:
            src = os.path.join(raw_dir, fname)
            dst = os.path.join(out_dir, fname)
            if not os.path.exists(dst):   # skip if a previous run already copied this file
                shutil.copy2(src, dst)
            copied += 1

        summary[species_name] = copied
        LOGGER("P05", f"  {species_name}: {len(raw_files)} total - {len(selected_files)} selected = {copied} remaining copied")

    return summary


# FUNCTION 19
def P05_F19_copyAllImagesForExcludedSpecies(excluded: list) -> dict:
    # for each excluded species (never had 200 images, never trained on),
    # copies ALL their images to excluded_species/<fish_XX>/
    # returns a dict {fish_XX: count_copied}

    summary           = {}
    excluded_out_root = os.path.join(M01_P05_OUTPUT_REMAINING_FOLDER, "excluded_species")

    for fish_folder in excluded:
        raw_dir = os.path.join(M01_INPUT_RAW_FISH_IMAGE_FOLDER, fish_folder)
        out_dir = os.path.join(excluded_out_root, fish_folder)
        os.makedirs(out_dir, exist_ok=True)

        all_files = sorted([
            f for f in os.listdir(raw_dir)
            if os.path.splitext(f)[1].lower() in _IMAGE_EXTS
        ])

        copied = 0
        for fname in all_files:
            src = os.path.join(raw_dir, fname)
            dst = os.path.join(out_dir, fname)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
            copied += 1

        summary[fish_folder] = copied
        LOGGER("P05", f"  {fish_folder}: {copied} images copied (all — excluded species)")

    return summary


# FUNCTION 20
def P05_F20_saveRemainingDatasetReport(
    known_summary:    dict,
    excluded_summary: dict,
) -> None:
    # writes a readable text report summarising how many images were copied and where

    os.makedirs(M01_OUTPUT_REPORTS_FOLDER, exist_ok=True)

    total_known    = sum(known_summary.values())
    total_excluded = sum(excluded_summary.values())
    total_all      = total_known + total_excluded

    lines = [
        "Path 4 V4 — M01_P05 Remaining Images Report",
        f"Generated: {datetime.utcnow().isoformat()}",
        "",
        "=" * 60,
        "KNOWN SPECIES (remaining after 200-image balanced selection)",
        "=" * 60,
    ]
    for species, count in sorted(known_summary.items()):
        lines.append(f"  {species:<40} {count:>6} images")
    lines.append(f"  {'SUBTOTAL':<40} {total_known:>6} images")

    lines += [
        "",
        "=" * 60,
        "EXCLUDED SPECIES (all images — never trained on)",
        "=" * 60,
    ]
    for folder, count in sorted(excluded_summary.items()):
        lines.append(f"  {folder:<40} {count:>6} images")
    lines.append(f"  {'SUBTOTAL':<40} {total_excluded:>6} images")

    lines += [
        "",
        "=" * 60,
        f"  {'GRAND TOTAL':<40} {total_all:>6} images",
        "=" * 60,
        "",
        f"Output folder: {M01_P05_OUTPUT_REMAINING_FOLDER}",
        "    known_species/    — remaining images from 10 trained species",
        "    excluded_species/ — all images from 13 untrained species",
        "",
        "Next step: run M07_P06 to bulk-predict all images in this folder.",
    ]

    with open(M01_P05_OUTPUT_REMAINING_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    LOGGER("P05", f"report saved -> {M01_P05_OUTPUT_REMAINING_REPORT_TXT}")


# MAIN
if __name__ == "__main__":
    LOGGER("P05", "=== M01_P05 — copying remaining images to analysis folder ===")

    LOGGER("P05", "step 1 — loading species map and classifying raw folders...")
    known_map, excluded = P05_F17_loadSpeciesMapAndIdentifyKnownAndExcludedFolders()

    LOGGER("P05", "step 2 — copying remaining images for known species...")
    known_summary = P05_F18_copyRemainingImagesForKnownSpecies(known_map)

    LOGGER("P05", "step 3 — copying all images for excluded species...")
    excluded_summary = P05_F19_copyAllImagesForExcludedSpecies(excluded)

    LOGGER("P05", "step 4 — saving report...")
    P05_F20_saveRemainingDatasetReport(known_summary, excluded_summary)

    total = sum(known_summary.values()) + sum(excluded_summary.values())
    LOGGER("P05", f"=== M01_P05 done — {total} images copied to {M01_P05_OUTPUT_REMAINING_FOLDER} ===")
    LOGGER("P05", "run M07_P06 next to bulk-predict all images in that folder")