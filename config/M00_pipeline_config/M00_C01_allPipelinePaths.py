# ========= FILE NAME: M00_C01_allPipelinePaths.py =========
# FILE ROLE: Single source of truth for every folder path in the project
# TOTAL FUNCTIONS IN THIS FILE: 0  (constants only, no functions)
# FUNCTION INDEX RANGE: none
# ALL THE FILE CONNECT TO (0): nothing — this file is imported by everyone else

import os

# ─── ROOT ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = r"D:\ASDF\Nam 4 HK 2\Đồ án tốt nghiệp\Path 4 V4"

PROJECT_ROOT_OUTPUTS = r"D:\ASDF\Nam 4 HK 2\Đồ án tốt nghiệp\Path 4 V4\outputs"


# ─── M01 DATASET ─────────────────────────────────────────────────────────────

M01_P02_INPUT_SPECIES_MAP      = os.path.join(PROJECT_ROOT, "semantic", "M05_semantic", "input", "M05_I02_speciesLabelMap.json")
M01_INPUT_RAW_FISH_IMAGE_FOLDER = r"D:\ASDF\Nam 4 HK 2\Đồ án tốt nghiệp\Path 4 V4\dataset\DATA DOWNLOAD\fish_image"

M01_P02_OUTPUT_BALANCED_FOLDER           = os.path.join(PROJECT_ROOT_OUTPUTS, "M01_dataset", "P02_balanced_200")

M01_P03_OUTPUT_TRAIN_FOLDER             = os.path.join(PROJECT_ROOT_OUTPUTS, "M01_dataset", "P03_split_140_30_30", "train")
M01_P03_OUTPUT_VAL_FOLDER                = os.path.join(PROJECT_ROOT_OUTPUTS, "M01_dataset", "P03_split_140_30_30", "val")
M01_P03_OUTPUT_TEST_FOLDER               = os.path.join(PROJECT_ROOT_OUTPUTS, "M01_dataset", "P03_split_140_30_30", "test")

M01_OUTPUT_REPORTS_FOLDER       = os.path.join(PROJECT_ROOT_OUTPUTS, "M01_dataset", "reports")

M01_P01_OUTPUT_EXPLORATION_CSV = os.path.join(M01_OUTPUT_REPORTS_FOLDER, "M01_P01_dataset_exploration.csv")

M01_P04_OUTPUT_DISTRIBUTION_PNG= os.path.join(M01_OUTPUT_REPORTS_FOLDER, "M01_P04_class_distribution.png")
M01_P04_OUTPUT_SUMMARY_MD      = os.path.join(M01_OUTPUT_REPORTS_FOLDER, "M01_P04_dataset_summary.md")

#(optional check)
M01_P05_OUTPUT_REMAINING_FOLDER          = os.path.join(PROJECT_ROOT_OUTPUTS, "M01_dataset", "P05_remaining_images")
M01_P05_OUTPUT_REMAINING_REPORT_TXT      = os.path.join(M01_OUTPUT_REPORTS_FOLDER, "M01_P05_remaining_images_report.txt")


# ─── M02 PREPROCESSING ───────────────────────────────────────────────────────

# inputs — read directly from M01 output, no copying
M02_P01_P02_INPUT_TRAIN_FOLDER          = M01_P03_OUTPUT_TRAIN_FOLDER


# outputs
M02_P01_OUTPUT_NORM_STATS_JSON      = os.path.join(PROJECT_ROOT_OUTPUTS, "M02_preprocessing", "M02_P01_P02_norm_stats.json")
#M02_OUTPUT_SAMPLE_GRID     = os.path.join(PROJECT_ROOT_OUTPUTS, "M02_preprocessing", "sample_augmented_grid.png")

# ─── M03 MODEL ───────────────────────────────────────────────────────────────

# inputs — read from M01 and M02 outputs
M03_P03_INPUT_TRAIN_FOLDER          = M01_P03_OUTPUT_TRAIN_FOLDER
M03_P03_INPUT_VAL_FOLDER            = M01_P03_OUTPUT_VAL_FOLDER
M03_P03_INPUT_TEST_FOLDER           = M01_P03_OUTPUT_TEST_FOLDER
M03_P03_INPUT_NORM_STATS_JSON       = M02_P01_OUTPUT_NORM_STATS_JSON


M03_P06_INPUT_NORM_STATS_JSON  = M02_P01_OUTPUT_NORM_STATS_JSON
M03_P06_INPUT_BALANCED_FOLDER   = M01_P02_OUTPUT_BALANCED_FOLDER

# outputs
M03_OUTPUT_WEIGHTS_FOLDER         = os.path.join(PROJECT_ROOT_OUTPUTS, "M03_model", "weights")
M03_OUTPUT_LOGS_FOLDER            = os.path.join(PROJECT_ROOT_OUTPUTS, "M03_model", "logs")

M03_P03_OUTPUT_CLASS_TO_IDX_JSON    = os.path.join(M03_OUTPUT_WEIGHTS_FOLDER, "M03_P03_class_to_idx.json")

M03_P05_OUTPUT_BEST_MODEL_PTH     = os.path.join(M03_OUTPUT_WEIGHTS_FOLDER, "M03_P05_best_model.pth")
M03_P05_OUTPUT_FINAL_MODEL_PTH    = os.path.join(M03_OUTPUT_WEIGHTS_FOLDER, "M03_P05_final_model.pth")
M03_P05_OUTPUT_TRAINING_LOG_CSV    = os.path.join(M03_OUTPUT_LOGS_FOLDER, "M03_P05_training_log_final.csv")

M03_P06_OUTPUT_KFOLD_LOG_CSV       = os.path.join(M03_OUTPUT_LOGS_FOLDER, "M03_P06_kfold_results.csv")



# ─── M04 EVALUATION ──────────────────────────────────────────────────────────

# inputs — read from M01, M02, M03 outputs
M04_P01_INPUT_TEST_FOLDER          = M01_P03_OUTPUT_TEST_FOLDER
M04_P01_INPUT_NORM_STATS_JSON      = M02_P01_OUTPUT_NORM_STATS_JSON
M04_P01_INPUT_BEST_MODEL_PTH       = M03_P05_OUTPUT_BEST_MODEL_PTH
M04_P01_INPUT_CLASS_TO_IDX_JSON    = M03_P03_OUTPUT_CLASS_TO_IDX_JSON

# outputs
M04_OUTPUT_FOLDER          = os.path.join(PROJECT_ROOT_OUTPUTS, "M04_evaluation")

M04_P01_OUTPUT_PREDICTIONS_CSV = os.path.join(M04_OUTPUT_FOLDER, "M04_P01_all_test_predictions.csv")

M04_P02_OUTPUT_PER_CLASS_CI_CSV    = os.path.join(M04_OUTPUT_FOLDER, "M04_P02_per_class_accuracy_with_ci.csv")
M04_P02_OUTPUT_F1_CSV          = os.path.join(M04_OUTPUT_FOLDER, "M04_P02_per_class_f1_precision_recall.csv")

M04_P03_OUTPUT_CONFUSION_PNG   = os.path.join(M04_OUTPUT_FOLDER, "M04_P03_confusion_matrix.png")

M04_P04_OUTPUT_PR_CURVE_PNG    = os.path.join(M04_OUTPUT_FOLDER, "M04_P04_precision_recall_curve.png")
M04_P04_OUTPUT_THRESHOLD_JSON  = os.path.join(M04_OUTPUT_FOLDER, "M04_P04_best_confidence_threshold.json")

M04_P05_OUTPUT_F1_BAR_PNG      = os.path.join(M04_OUTPUT_FOLDER, "M04_P05_per_class_f1_bar_chart.png")
M04_P05_OUTPUT_MACRO_F1_JSON   = os.path.join(M04_OUTPUT_FOLDER, "M04_P05_macro_f1_summary.json")

# ─── M05 SEMANTIC ────────────────────────────────────────────────────────────

# inputs
M05_I02_INPUT_SPECIESLABELMAP_JSON   = os.path.join(PROJECT_ROOT, "semantic", "M05_semantic", "input", "M05_I02_speciesLabelMap.json")

M05_P03_INPUT_THRESHOLD              = M04_P04_OUTPUT_THRESHOLD_JSON


M05_P05_INPUT_TEST_FOLDER            = M01_P03_OUTPUT_TEST_FOLDER
M05_P05_INPUT_BEST_MODEL_PTH         = M03_P05_OUTPUT_BEST_MODEL_PTH
M05_P05_INPUT_CLASS_TO_IDX_JSON      = M03_P03_OUTPUT_CLASS_TO_IDX_JSON
M05_P05_INPUT_NORM_STATS_JSON        = M02_P01_OUTPUT_NORM_STATS_JSON

# outputs
M05_OUTPUT_FOLDER                    = os.path.join(PROJECT_ROOT_OUTPUTS, "M05_semantic")

M05_P01_OUTPUT_LABEL_SCIENTIFIC_JSON = os.path.join(M05_OUTPUT_FOLDER, "M05_P01_label_to_scientific.json")

M05_P02_OUTPUT_TAXONOMY_MASTER_JSON  = os.path.join(M05_OUTPUT_FOLDER, "M05_P02_taxonomy_master.json")

M05_P05_OUTPUT_INFERENCE_RESULTS_FOLDER  = os.path.join(M05_OUTPUT_FOLDER, "P05_inference_results")

M05_P05_OUTPUT_RUN_TXT         = os.path.join(M05_OUTPUT_FOLDER, "M05_P05_inference_run.txt")

# ─── M06 KNOWLEDGE GRAPH ─────────────────────────────────────────────────────

# inputs — read from M05 output
M06_P03_INPUT_INFERENCE_RESULTS_FOLDER  = M05_P05_OUTPUT_INFERENCE_RESULTS_FOLDER

# outputs
M06_P04_OUTPUT_QUERY_RESULTS_FOLDER   = os.path.join(PROJECT_ROOT_OUTPUTS, "M06_knowledge_graph", "P04_query_results")

# ─── M07 WEB APP ─────────────────────────────────────────────────────────────

# inputs — loads model and stats at startup
M07_P01_INPUT_BEST_MODEL_PTH       = M03_P05_OUTPUT_BEST_MODEL_PTH
M07_P01_INPUT_CLASS_TO_IDX_JSON    = M03_P03_OUTPUT_CLASS_TO_IDX_JSON
M07_P01_INPUT_NORM_STATS_JSON      = M02_P01_OUTPUT_NORM_STATS_JSON
M07_P01_INPUT_TAXONOMY_MASTER_JSON = M05_P02_OUTPUT_TAXONOMY_MASTER_JSON
M07_P01_INPUT_THRESHOLD_JSON       = M04_P04_OUTPUT_THRESHOLD_JSON

# outputs
M07_P01_OUTPUT_TEMPLATES_FOLDER       = os.path.join(PROJECT_ROOT_OUTPUTS, "M07_web_app", "output", "P01_templates")
M07_P01_OUTPUT_STATIC_FOLDER                 = os.path.join(PROJECT_ROOT_OUTPUTS, "M07_web_app", "output", "P01_static")

M07_P02_OUTPUT_UPLOADS_FOLDER         = os.path.join(PROJECT_ROOT_OUTPUTS, "M07_web_app", "output", "P02_uploads")


#(optional check)
M07_P06_OUTPUT_BULK_PREDICT_FOLDER       = os.path.join(PROJECT_ROOT_OUTPUTS, "M07_web_app", "P06_bulk_predict")
M07_P06_OUTPUT_BULK_PREDICTIONS_CSV      = os.path.join(M07_P06_OUTPUT_BULK_PREDICT_FOLDER, "M07_P06_bulk_predictions.csv")
