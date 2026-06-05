# ========= FILE NAME: M05_I01_inputPaths.py =========
# FILE ROLE: Tells M05 where its inputs and outputs live — reads from M00, adds nothing new
# TOTAL FUNCTIONS IN THIS FILE: 0  (re-exports only)
# FUNCTION INDEX RANGE: none

# M05_P01
# import the species label map input path and the label→scientific output path from the central config
from config.M00_pipeline_config.M00_C01_allPipelinePaths import (
    M05_I02_INPUT_SPECIESLABELMAP_JSON,   # JSON mapping internal fish folder labels to species names
    M05_P01_OUTPUT_LABEL_SCIENTIFIC_JSON  # JSON output for label → scientific-name lookup
)

#----------------------------------------------------------------#

# M05_P02
# import the taxonomy master JSON output path from the central config
from config.M00_pipeline_config.M00_C01_allPipelinePaths import (
    M05_P02_OUTPUT_TAXONOMY_MASTER_JSON   # full WoRMS taxonomy cache for all species
)

# reuse the P01 output as the P02 input so taxonomy building starts from the label→scientific map
M05_P02_INPUT_LABEL_SCIENTIFIC_JSON = M05_P01_OUTPUT_LABEL_SCIENTIFIC_JSON

#----------------------------------------------------------------#

# M05_P03
# import the threshold JSON path used to decide whether a prediction is confident enough
from config.M00_pipeline_config.M00_C01_allPipelinePaths import (
    M05_P03_INPUT_THRESHOLD   # confidence threshold saved by M04_P04
)

# assign the taxonomy master path as the input for the enrichment step
# note: this variable name starts with M03_ in the source, which is likely a typo
M03_P03_INPUT_TAXONOMY_MASTER_JSON = M05_P02_OUTPUT_TAXONOMY_MASTER_JSON

#----------------------------------------------------------------#

# M05_P05
# import the inference inputs and outputs used by the full semantic pipeline
from config.M00_pipeline_config.M00_C01_allPipelinePaths import (
    M05_P05_INPUT_TEST_FOLDER,           # test image folder from M01 split
    M05_P05_INPUT_BEST_MODEL_PTH,        # trained best model weights from M03
    M05_P05_INPUT_CLASS_TO_IDX_JSON,     # class name ↔ numeric index mapping
    M05_P05_INPUT_NORM_STATS_JSON,       # normalization stats from M02

    M05_P05_OUTPUT_INFERENCE_RESULTS_FOLDER,  # folder for one JSON file per inference result
    M05_OUTPUT_FOLDER,                         # root output folder for M05
    M05_P05_OUTPUT_RUN_TXT                     # log text file for the inference run
)

#----------------------------------------------------------------#

# M05_P06
# import evaluation outputs from M04 and the test folder from M01 for error analysis
from config.M00_pipeline_config.M00_C01_allPipelinePaths import (
    M04_P01_OUTPUT_PREDICTIONS_CSV,   # predictions CSV from M04_P01
    M01_P03_OUTPUT_TEST_FOLDER,       # exact test folder from M01
    M04_P04_OUTPUT_THRESHOLD_JSON,    # confidence threshold JSON from M04_P04
    M04_OUTPUT_FOLDER,                # root output folder for M04
)

#----------------------------------------------------------------#

# import settings used by WoRMS and M05 inference
from config.M00_pipeline_config.M00_C02_allSettings import (
    M05_P02_WORMS_API_SLEEP_SECONDS,  # delay between WoRMS API calls
    M05_P02_WORMS_BASE_URL,           # WoRMS REST base URL

    M02_P01_P02_INPUT_IMAGE_SIZE as M05_P05_INPUT_IMAGE_SIZE  # image size reused for inference transforms
)

# import shared logger functions
from config.M00_pipeline_config.M00_C03_logger import (
    LG00_F01_printStepToConsole as LOGGER,       # normal progress logger
    LG00_F02_printWarningToConsole as WARN_LOGGER,  # warning logger
)

