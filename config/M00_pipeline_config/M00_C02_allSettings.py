# ========= FILE NAME: M00_C02_allSettings.py =========
# FILE ROLE: Single source of truth for every number and setting in the project
# TOTAL FUNCTIONS IN THIS FILE: 0  (constants only, no functions)
# FUNCTION INDEX RANGE: none
# ALL THE FILE CONNECT TO (0): nothing — imported by anyone who needs a setting

# ─── DATASET ─────────────────────────────────────────────────────────────────

M01_P02_MIN_IMAGES_PER_SPECIES = 200    # species with fewer images than this get dropped
M01_P02_SAMPLE_SIZE_PER_SPECIES = 200   # how many images to keep per species after balancing

M01_P03_TRAIN_COUNT = 140               # images per species in the training pile
M01_P03_VAL_COUNT   = 30                # images per species in the validation pile
M01_P03_TEST_COUNT  = 30                # images per species in the test pile

RANDOM_SEED = 20                # used everywhere random choices are made

# ─── PREPROCESSING ───────────────────────────────────────────────────────────

M02_P01_P02_INPUT_IMAGE_SIZE = 224          # ResNet-18 expects exactly 224×224 pixels

# ─── MODEL ───────────────────────────────────────────────────────────────────
M03_INPUT_IMAGE_SIZE = M02_P01_P02_INPUT_IMAGE_SIZE

M03_NUM_CLASSES    = 10             # number of fish species the model learns
M03_BATCH_SIZE     = 16             # how many photos the model sees at once
M03_LEARNING_RATE  = 0.001          # how big each learning step is
M03_NUM_EPOCHS     = 30             # maximum number of full practice rounds
M03_PATIENCE       = 5              # stop early if no improvement after this many rounds
M03_KFOLD_K        = 5              # how many times to re-run training in k-fold check

# ─── EVALUATION ──────────────────────────────────────────────────────────────

M04_CI_CONFIDENCE_LEVEL = 0.95      # confidence interval width (95%)
# NOTE: the actual confidence threshold used for predictions is NOT set here.
# It is derived from the precision-recall curve in M04 and saved to
# M04_OUTPUT_THRESHOLD_JSON. Never hardcode a threshold number here.

# ─── WoRMS API ───────────────────────────────────────────────────────────────

M05_P02_WORMS_API_SLEEP_SECONDS = 1     # wait this long between WoRMS API calls
M05_P02_WORMS_BASE_URL = "https://www.marinespecies.org/rest"

# ─── NEO4J ───────────────────────────────────────────────────────────────────

M06_P01_NEO4J_URI      = "neo4j://127.0.0.1:7687"
M06_P01_NEO4J_USER     = "neo4j"
M06_P01_NEO4J_PASSWORD = "123456789"
M06_NEO4J_DATABASE = "neo4j"

# ─── FLASK ───────────────────────────────────────────────────────────────────

M07_FLASK_HOST          = "127.0.0.1"
M07_FLASK_PORT          = 5000
M07_FLASK_DEBUG         = False
M07_FLASK_SECRET_KEY    = "1234"
M07_UPLOAD_MAX_AGE_MINS = 10        # uploaded images are deleted after this many minutes