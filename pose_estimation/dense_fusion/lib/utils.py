"""Utils
"""

import logging
import torch
from torch.autograd import Function
from vision_sandbox import _C as knn_pytorch

def setup_logger(logger_name, log_file, level=logging.WARN):
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
    fileHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)

    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    l.addHandler(streamHandler)

    return l


class KNearestNeighbor(Function):
  """ Compute k nearest neighbors for each query point.
  """
  def __init__(self, k):
    self.k = k

  def forward(self, ref, query):
    ref = ref.float().cuda()
    query = query.float().cuda()

    inds = torch.empty(query.shape[0], self.k, query.shape[2]).long().cuda()

    knn_pytorch.knn(ref, query, inds)

    return inds