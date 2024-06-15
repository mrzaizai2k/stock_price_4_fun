import sys
sys.path.append("")

import logging
from logging.handlers import TimedRotatingFileHandler

def create_logger(logfile='logs/facesvc.log'):
    logger = logging.getLogger()
    # Set the logging level
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    rotate_handler = TimedRotatingFileHandler(filename=logfile, when="midnight", backupCount=30)
    rotate_handler.setLevel(logging.DEBUG)
    rotate_handler.suffix = "%Y%m%d"
    
    formatter = logging.Formatter('%(asctime)s - %(filename)s:%(lineno)s - %(levelname)s - %(message)s')
    rotate_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(rotate_handler)
    return logger
