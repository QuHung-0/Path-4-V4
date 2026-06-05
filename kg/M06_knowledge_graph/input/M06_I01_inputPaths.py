# ========= FILE NAME: M06_I01_inputPaths.py =========
# FILE ROLE: Tells M06 where its inputs live — reads from M00, adds nothing new
# TOTAL FUNCTIONS IN THIS FILE: 0  (re-exports only)
# FUNCTION INDEX RANGE: none

# M04_P01
# import Neo4j connection settings from the central settings file
from config.M00_pipeline_config.M00_C02_allSettings import (
    M06_P01_NEO4J_URI,       # Neo4j connection URI
    M06_P01_NEO4J_USER,      # Neo4j username
    M06_P01_NEO4J_PASSWORD,  # Neo4j password
)

#----------------------------------------------------------------#

# M04_P03
# import the folder containing all JSON inference results written by M05
from config.M00_pipeline_config.M00_C01_allPipelinePaths import (
    M06_P03_INPUT_INFERENCE_RESULTS_FOLDER,  # folder of JSON records from M05_P05
)

#----------------------------------------------------------------#

# M04_P04
# import the folder where query results will be written
from config.M00_pipeline_config.M00_C01_allPipelinePaths import (
    M06_P04_OUTPUT_QUERY_RESULTS_FOLDER,  # output folder for CSV query results
)

#----------------------------------------------------------------#

# import the Neo4j database name from settings
from config.M00_pipeline_config.M00_C02_allSettings import (
    M06_NEO4J_DATABASE,  # target Neo4j database name
)

# import shared loggers used by M06 scripts
from config.M00_pipeline_config.M00_C03_logger import (
    LG00_F01_printStepToConsole as LOGGER,      # normal progress logger
    LG00_F02_printWarningToConsole as WARN_LOGGER,  # warning logger
)