import os
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger
from config import LOG_FILE_PATH

def get_logger(name, filename=None):
    if filename == None:
        filename = LOG_FILE_PATH

    logger = getLogger(name)

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    handler = FileHandler(filename=filename)
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    return logger
