# ========= FILE NAME: M00_C03_logger.py =========
# FILE ROLE: Prints clearly labelled status messages so you always know what is running
# TOTAL FUNCTIONS IN THIS FILE: 2
# FUNCTION INDEX RANGE: LG00_F01 → LG00_F02
# ALL THE FILE CONNECT TO (0): nothing — imported by any process script that needs to print

from datetime import datetime

def LG00_F01_printStepToConsole(module_name: str, message: str) -> None:
    # prints a timestamped info line so you can track progress
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{module_name}] {message}")


def LG00_F02_printWarningToConsole(module_name: str, message: str) -> None:
    # prints a timestamped warning line so problems stand out clearly
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{module_name}] WARNING: {message}")