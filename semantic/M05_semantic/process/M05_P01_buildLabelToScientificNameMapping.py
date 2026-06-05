# ========= FILE NAME: M05_P01_buildLabelToScientificNameMapping.py =========
# FILE ROLE: Reads the species label map JSON and saves a clean label→scientific name lookup
# TOTAL FUNCTIONS IN THIS FILE: 3

# import OS utilities for folder creation
import os

# import JSON for reading and writing mapping files
import json

# import the shared input paths and logger for M05
from semantic.M05_semantic.input.M05_I01_inputPaths import (
    M05_I02_INPUT_SPECIESLABELMAP_JSON,   # raw label map JSON input
    M05_P01_OUTPUT_LABEL_SCIENTIFIC_JSON, # output JSON for label→scientific mapping
    LOGGER                                # logger for progress messages
)

# FUNCTION 1
def P01_F01_readSpeciesLabelMapFromJsonFile(map_json_path: str) -> dict:
    # loads {fish_XX: species_name} from the manually created label map JSON
    with open(map_json_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)  # read the JSON file into a Python dictionary
    LOGGER("P01", f"label map loaded — {len(label_map)} entries from {map_json_path}")
    return label_map

# FUNCTION 2
def P01_F02_buildInternalLabelToScientificNameLookup(label_map: dict) -> dict:
    # inverts the label map into a cleaner lookup for later use
    # the folder names use underscores, so this replaces underscores with spaces
    lookup = {}
    for folder_id, species_name in label_map.items():
        scientific = species_name.replace("_", " ")   # convert folder-style name to readable scientific-style text
        lookup[species_name] = scientific             # store: internal label → spaced scientific name
        LOGGER("P01", f"  {species_name}  →  {scientific}")
    return lookup

# FUNCTION 3
def P01_F03_saveLabelToScientificNameMapToJsonFile(lookup: dict, output_path: str) -> None:
    # saves the lookup dict to a JSON file so other modules can use it without rebuilding it
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # create parent folder if needed
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(lookup, f, indent=4, sort_keys=True)  # write pretty JSON with sorted keys
    LOGGER("P01", f"label→scientific map saved → {output_path}")

# MAIN
if __name__ == "__main__":
    LOGGER("P01", "=== M05_P01 — building label to scientific name mapping ===")

    LOGGER("P01", "step 1 — reading species label map...")
    label_map = P01_F01_readSpeciesLabelMapFromJsonFile(M05_I02_INPUT_SPECIESLABELMAP_JSON)

    LOGGER("P01", "step 2 — building internal label → scientific name lookup...")
    lookup = P01_F02_buildInternalLabelToScientificNameLookup(label_map)

    LOGGER("P01", "step 3 — saving lookup to file...")
    P01_F03_saveLabelToScientificNameMapToJsonFile(lookup, M05_P01_OUTPUT_LABEL_SCIENTIFIC_JSON)

    LOGGER("P01", "=== M05_P01 done ===")