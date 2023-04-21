# -*- coding: utf-8 -*-
# @Time         : 2023/4/21 15:34
# @Author       : Yupeng Ji
# @File         : split_train.py
# @Description  :
"""
原来的eval_dataset.txt 无label, 只有train_dataset.txt。因此将train_dataset.txt按照8：2分割成
splited_train_data.txt 与 splited_eval_data.txt
"""

import os
import random

with open('train_dataset.txt', 'r') as f:
    raw_train_dataset = f.readlines()
raw_train_dataset = [dt.rstrip('\n') for dt in raw_train_dataset]

data_length = len(raw_train_dataset)
idx = list(range(0, data_length))
random.shuffle(idx)

split_ratio = 0.8
train_idx = idx[:int(data_length * split_ratio)]
eval_idx = idx[int(data_length * split_ratio): ]

with open('splited_train_data.txt', 'w') as f:
    for i in train_idx:
        f.write(f"{raw_train_dataset[i]}\n")

with open('splited_eval_data.txt', 'w') as f:
    for i in eval_idx:
        f.write(f"{raw_train_dataset[i]}\n")


