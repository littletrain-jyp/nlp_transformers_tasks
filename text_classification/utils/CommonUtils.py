# -*- coding: utf-8 -*-
# @Time         : 2023/4/13 19:03
# @Author       : Yupeng Ji
# @File         : CommonUtils.py
# @Description  :

import random
import os
import json
import logging
import time
import pickle
import numpy as np
import torch

def set_seed(seed=2023):
    """
    设置随机数种子，保证实验可重现
    :param seed:
    :return:
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_logger(log_path):
    """
    配置logger
    :param log_path
    :return:
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not any(handler.__class__ == logging.FileHandler for handler in logger.handlers): # 输出到文件
        file_handler = logging.FileHandler(log_path)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)d - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if not any(handler.__class__ == logging.StreamHandler for handler in logger.handlers): # StreamHandler 输出到控制台
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

