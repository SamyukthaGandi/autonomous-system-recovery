import logging
import os

# Define log directory and file
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "system.log")

# Ensure the log directory exists
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Configure the logger
logging.basicConfig(
    level=logging.INFO,  # Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),  # Log to file
        logging.StreamHandler()  # Log to console
    ]
)

# Logger instance
logger = logging.getLogger("SystemLogger")

def log_info(message):
    """Log an informational message."""
    logger.info(message)

def log_warning(message):
    """Log a warning message."""
    logger.warning(message)

def log_error(message):
    """Log an error message."""
    logger.error(message)

def log_critical(message):
    """Log a critical error message."""
    logger.critical(message)

# Example usage
if __name__ == "__main__":
    log_info("System started successfully.")
    log_warning("High memory usage detected.")
    log_error("Failed to connect to database.")
    log_critical("System crash detected!")
