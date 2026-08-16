"""
#############################################################################
### Custom logger file
###
### @file custom_logger.py
### @author Sebastian Russo
### @date 2025
#############################################################################

This module initializes a custom logger to handle log messages for the other modules.
"""

#Native imports
import os
import logging
import sys
import datetime

# --- CONFIGURATION AREA ---
#Map config string levels to logging module levels
LOG_LEVELS = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
    "notset": logging.NOTSET
}
LOG_FILE_NAME = "log_file_name_should_come_from_a_config_file_in_lower_case"
LOG_LEVEL_STR = "logging_level_should_come_from_a_config_file_in_lower_case"
LOG_DIRECTORY = "log_directory_name_should_come_from_a_config_file_in_lower_case"

#Get log level string
log_level = LOG_LEVELS.get(LOG_LEVEL_STR.lower(), logging.INFO)

# --- Log basic configuration and formatting ---
log_handler = logging.getLogger(LOG_FILE_NAME)
log_handler.setLevel(log_level)

# --- Logger formatter ---
log_format = logging.Formatter(
    fmt="%(asctime)s %(msecs)03dZ | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# --- File handler initialization ---
file_handler = None
log_file = None

try:
    #Create folder
    log_directory = LOG_DIRECTORY
    os.makedirs(log_directory, exist_ok=True)
    
    #Create log file
    log_file = os.path.join(
                        log_directory, 
                        datetime.datetime.now().strftime(
                            f"{LOG_FILE_NAME}_%Y-%m-%dT%H-%M-%S.log"))
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(log_format)
except OSError as e:
    sys.stderr.write(f"ERROR: Failed to create log file at '{log_file}': {e}\n")
    sys.stderr.write("Continuing with console-only logging.\n")
    file_handler = None
except Exception as e:
    sys.stderr.write(f"ERROR: Unexpected error during log file initialization: {e}\n")
    sys.stderr.write("Continuing with console-only logging.\n")
    file_handler = None

#Console handler to logs
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_format)

#Final log handler
if not log_handler.hasHandlers():
    if file_handler is not None:
        log_handler.addHandler(file_handler)
    log_handler.addHandler(console_handler)

log_handler.info("Project name backend server starting")
if log_file:
    log_handler.warning(f"Current working directory: {os.getcwd()}, Logs are written to "
                        f"'{log_file}'")
else:
    log_handler.warning(f"Current working directory: {os.getcwd()}, File logging "
                        f"unavailable - console only")

# --- Shutdown function ---
def shutdown_logger():
    """
    Properly closes and flushes all log handlers.
    Call this before application exit to ensure all logs are written.
    (Optional: Python's garbage collection will handle it, but explicit is better.)
    """
    try:
        for handler in log_handler.handlers[:]:
            try:
                handler.flush()
                handler.close()
                log_handler.removeHandler(handler)
            except Exception as e:
                sys.stderr.write(f"Error closing log handler: {e}\n")
    except Exception as e:
        sys.stderr.write(f"Error during logger shutdown: {e}\n")

#Example usage
"""
from src.utils.custom_logger import log_handler

log_handler.debug("Debug message")
log_handler.info("Info message")
log_handler.warning("Warning message")
log_handler.error("Error message")
log_handler.critical("Critical message")

# Call before program exit (optional):
# shutdown_logger()
"""
