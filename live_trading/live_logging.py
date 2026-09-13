"""
logging for trades/errors 
"""
import os
import logging
from datetime import datetime
import json

def logger_setup(root_path: str, 
                 name: str, 
                 level: logging = logging.INFO, 
                 formatting: bool = False):
    """instentiates loggers, and creates folder/files. Formatting true loggers intended for debuging/warning/error"""
    # check directory exists
    os.makedirs(root_path, exist_ok = True) # create logs folder
    save_path = os.path.join(root_path, name)

    logger = logging.getLogger(name) # module name when imported 
    logger.setLevel(level)
    handler = logging.FileHandler(filename = save_path)
    if formatting:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s") # message required parameter, by calling logger.dubug/info/warning/...
        )
    logger.addHandler(handler)

    return logger

def trades_logger(trades_logger: str, 
                  event: str, 
                  **kwargs): 
    """writes logging dict, custom formatting"""
    payload = {'timestamp': datetime.now().isoformat(), **kwargs}
    return logging.getLogger(trades_logger).info(f"{event}: %s", json.dumps(payload, default = str))