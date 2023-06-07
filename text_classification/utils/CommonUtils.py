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
from datetime import timedelta

logger = logging.getLogger(__name__)

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

def print_run_time(func):
    """时间装饰器"""

    def wrapper(*args, **kw):
        local_time = time.time()
        func(*args, **kw)
        logging.info(f'[file:{__file__}][{func.__name__}] run time is {time.time() - local_time :4f}')

    return wrapper

def get_time_diff(start_time):
    """ 获取已使用的时间"""
    end_time = time.time()
    time_diff = end_time - start_time
    return timedelta(seconds=int(round(time_diff)))

def keep_checkpoints_num(ckpt_dir, ckpt_filename_keyword='checkpoint', ckpt_nums=3):
    """
    维护ckpt_dir文件夹里checkpoints的数量
    :param ckpt_dir: 存储checkpoints的文件夹
    :param ckpt_filename_keyword: 单个checkpoint文件夹名字所包含的keyword，用来挑选出ckpt的文件夹名称
    :param ckpt_nums: 存储ckpt的数量
    :return:
    """
    import shutil
    from pathlib import Path

    def get_ckpt_dirlist(ckpt_dir, ckpt_filename_keyword):
        ckpt_dirlist = []
        # 找到所有ckpt的文件夹名
        for ckpt in Path(ckpt_dir).iterdir():
            if ckpt_filename_keyword in ckpt.name:
                ckpt_dirlist.append(ckpt)
        return ckpt_dirlist

    ckpt_dirlist = get_ckpt_dirlist(ckpt_dir, ckpt_filename_keyword)
    # 按照生成时间进行排序，降序
    ckpt_dirlist = sorted(ckpt_dirlist, key = lambda ckpt: ckpt.stat().st_mtime, reverse=True)

    # 移除不满足条件的ckpt
    remove_ckpt_dirlist = ckpt_dirlist[ckpt_nums - 1:]
    for remove_ckpt in remove_ckpt_dirlist:
        shutil.rmtree(remove_ckpt)
        logging.info(f"{remove_ckpt} is removed!")
    return

def get_latest_checkpoints_dir(ckpt_dir, ckpt_filename_keyword='checkpoint'):
    """
    维护ckpt_dir文件夹里checkpoints的数量
    :param ckpt_dir: 存储checkpoints的文件夹
    :param ckpt_filename_keyword: 单个checkpoint文件夹名字所包含的keyword，用来挑选出ckpt的文件夹名称
    :param ckpt_nums: 存储ckpt的数量
    :return:
    """
    import shutil
    from pathlib import Path

    def get_ckpt_dirlist(ckpt_dir, ckpt_filename_keyword):
        ckpt_dirlist = []
        # 找到所有ckpt的文件夹名
        for ckpt in Path(ckpt_dir).iterdir():
            if ckpt_filename_keyword in ckpt.name:
                ckpt_dirlist.append(ckpt)
        return ckpt_dirlist

    ckpt_dirlist = get_ckpt_dirlist(ckpt_dir, ckpt_filename_keyword)
    if ckpt_dirlist:
        # 按照生成时间进行排序，降序
        ckpt_dirlist = sorted(ckpt_dirlist, key = lambda ckpt: ckpt.stat().st_mtime, reverse=True)
        return ckpt_dirlist[0]
    else:
        return ckpt_dirlist

def get_multi_onehot_label(labels, label_size, dtype=torch.long):
    """ 针对输入label，生成multi_onehot label
    e.g. input: [[0,1], [2]]
         output: [[1,1,0], [0,0,1]]
    """
    batch_size = len(labels)
    max_label_num = max([len(x) for x in labels])
    labels_extend = [[labels[i][0] for x in range(max_label_num)] for i in range(batch_size)]

    for i in range(0, batch_size):
        labels_extend[i][0: len(labels[i])] = labels[i]

    y = torch.Tensor(labels_extend).long()
    y_onehot = torch.zeros(batch_size, label_size, dtype=dtype).scatter_(1, y, 1)
    return y_onehot

def load_ckp(model, optimizer, checkpoint_path, device='cpu'):
    """ 读取ckpt"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.to(device)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    global_step = checkpoint['global_step']
    return model, optimizer, epoch, loss, global_step

def load_model(model, checkpoint_path, device='cpu'):
    checkpoint = torch.load(checkpoint_path, device)
    model.load_state_dict(checkpoint['state_dict'])
    logger.info(f"---{checkpoint_path} read successfully! ")
    return model

def save_ckp(state, checkpoint_path):
    """ 保存ckpt"""
    torch.save(state, checkpoint_path)
    logger.info(f"---{checkpoint_path} saved successfully! ")


if __name__ == "__main__":
    print(get_multi_onehot_label([[0,1], [2], [3,2]], 4))