# ========= FILE NAME: M04_I01_inputPaths.py =========
# FILE ROLE: Tells M04 where its inputs live — reads from M00, adds nothing new
# TOTAL FUNCTIONS IN THIS FILE: 0  (re-exports only)
# FUNCTION INDEX RANGE: none

# M04_P01
# import M04 input/output path constants from the central pipeline config
from config.M00_pipeline_config.M00_C01_allPipelinePaths import (
    M04_P01_INPUT_TEST_FOLDER,          # test image folder from M01 split
    M04_P01_INPUT_NORM_STATS_JSON,      # normalization stats JSON from M02
    M04_P01_INPUT_BEST_MODEL_PTH,       # best trained model weights from M03
    M04_P01_INPUT_CLASS_TO_IDX_JSON,    # class-to-index mapping from M03

    M04_P01_OUTPUT_PREDICTIONS_CSV      # output CSV containing all test predictions
)

# import model/dataset helpers reused by M04_P01
from model_weights.M03_model.process.M03_P03_buildDataloadersForTrainingAndValidation import (
    P03_F08_createImageFolderDatasetFromPath,            # create ImageFolder dataset
    P03_F10_wrapDatasetInDataloaderWithBatchSizeAndShuffle,  # wrap dataset in DataLoader
    P03_F11_loadClassToIndexMappingFromJsonFile,         # load idx→class mapping
)
from model_weights.M03_model.process.M03_P07_loadModelAndPredictOneSingleImage import (
    P07_F23_loadTrainedModelWeightsFromFile             # load saved trained model
)
from dataset.M02_preprocessing.process.M02_P02_defineAllTransformPipelines import (
    P02_F04_loadNormalizationStatsFromJsonFile,         # load mean/std from JSON
    P02_F06_buildValAndTestTransformNoAugmentation,     # val/test preprocessing pipeline
)

#----------------------------------------------------------------#

# M04_P02
# import M04 output paths for per-class metrics CSVs
from config.M00_pipeline_config.M00_C01_allPipelinePaths import (
    M04_P02_OUTPUT_PER_CLASS_CI_CSV,    # per-class Wilson confidence interval CSV
    M04_P02_OUTPUT_F1_CSV               # per-class precision/recall/F1 CSV
)

#----------------------------------------------------------------#

# M04_P03
# import M04 output path for confusion matrix image
from config.M00_pipeline_config.M00_C01_allPipelinePaths import (
    M04_P03_OUTPUT_CONFUSION_PNG        # confusion matrix heatmap PNG
)

#----------------------------------------------------------------#

# M04_P04
# import M04 output paths for precision-recall curve and threshold JSON
from config.M00_pipeline_config.M00_C01_allPipelinePaths import (
    M04_P04_OUTPUT_PR_CURVE_PNG,        # precision/recall/F1 curve PNG
    M04_P04_OUTPUT_THRESHOLD_JSON       # best confidence threshold JSON
)

#----------------------------------------------------------------#

# M04_P05
# import M04 output paths for F1 bar chart and macro-F1 summary JSON
from config.M00_pipeline_config.M00_C01_allPipelinePaths import (
    M04_P05_OUTPUT_F1_BAR_PNG,          # per-class F1 bar chart PNG
    M04_P05_OUTPUT_MACRO_F1_JSON,       # macro-F1 summary JSON
    M04_OUTPUT_FOLDER                   # root M04 evaluation output folder
)

#----------------------------------------------------------------#

# import numeric settings for M04
from config.M00_pipeline_config.M00_C02_allSettings import (
    M03_BATCH_SIZE as M04_BATCH_SIZE,           # reuse model batch size for test inference
    M02_P01_P02_INPUT_IMAGE_SIZE as M04_INPUT_IMAGE_SIZE,  # reuse preprocessing image size
    M04_CI_CONFIDENCE_LEVEL                     # confidence level for Wilson intervals
)

# import logger utilities
from config.M00_pipeline_config.M00_C03_logger import (
    LG00_F01_printStepToConsole as LOGGER,      # normal progress logger
    LG00_F02_printWarningToConsole as WARN_LOGGER,  # warning logger
)