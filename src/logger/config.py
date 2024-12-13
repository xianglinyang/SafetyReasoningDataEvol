import logging
import os
import sys
import time

def setup_logging(
    task_name, # train, eval, etc.
    log_level=logging.INFO,
    log_dir="logs",
    run_id=None,
):

    # Create logs directory if it doesn't exist
    log_dir = os.path.join(log_dir, task_name)
    os.makedirs(log_dir, exist_ok=True)

    # Create log file path with timestamp
    time_stamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(log_dir, f"{time_stamp}_{run_id}.log") if run_id else os.path.join(log_dir, f"{time_stamp}.log")

    # Setup logging format
    log_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    date_format = "%m/%d/%Y %H:%M:%S"
    
    # Setup logging to both file and console
    logging.basicConfig(
        format=log_format,
        datefmt=date_format,
        level=log_level,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


if __name__ == "__main__":
    setup_logging("test")
    logger = logging.getLogger(__name__)
    logger.info("This is a test log message")
