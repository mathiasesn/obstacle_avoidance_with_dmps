"""Utils for DenseFusion
"""

import logging

def setup_logger(logger_name, log_file, level=logging.INFO):
    """Setup logger
    
    Arguments:
        logger_name {[type]} -- [description]
        log_file {[type]} -- [description]
    
    Keyword Arguments:
        level {[type]} -- [description] (default: {logging.INFO})
    
    Returns:
        Logger -- [description]
    """
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')

    l.setLevel(level)
    l.addHandler(fileHandler)

    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    l.addHandler(streamHandler)

    return l