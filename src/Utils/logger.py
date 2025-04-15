import sys
sys.path.append("")

import logging
from logging.handlers import TimedRotatingFileHandler

import logging
from logging.handlers import TimedRotatingFileHandler
import os

def create_logger(logfile='logging/stock_bot.log'):
    # Create logger
    logger = logging.getLogger('stock_bot')
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # Prevent log duplication

    # Only add handlers if none exist
    if not logger.handlers:
        # Ensure directory exists
        os.makedirs(os.path.dirname(logfile), exist_ok=True)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create rotating file handler
        rotate_handler = TimedRotatingFileHandler(
            filename=logfile,
            when="midnight",
            backupCount=5
        )
        rotate_handler.setLevel(logging.DEBUG)
        rotate_handler.suffix = "%Y%m%d"

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] - Message: %(message)s'
        )

        # Set formatters
        console_handler.setFormatter(formatter)
        rotate_handler.setFormatter(formatter)

        # Add handlers
        logger.addHandler(console_handler)
        logger.addHandler(rotate_handler)

    return logger