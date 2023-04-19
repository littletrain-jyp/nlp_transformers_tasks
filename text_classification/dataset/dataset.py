# -*- coding: utf-8 -*-
# @Time         : 2023/4/19 19:53
# @Author       : Yupeng Ji
# @File         : dataset.py
# @Description  :

import os
import json
import torch
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

class ClassificationDataset(Dataset):
    """
    基类，直接传入文件名或者list形式的数据都可以
    """
    def __init__(self, file_path=None, data=None, **kwargs):
        self.kwargs = kwargs
        if isinstance(file_path, (str, list)):
            self.data = self.load_data(file_path)
        elif isinstance(data, list):
            self.data = data
        else:
            raise ValueError ("The input args shall be str format file_path / list format dataset")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    @staticmethod
    def load_data(file_path):
        return file_path

# 自定义处理相关数据
class CnewsDataset(ClassificationDataset):
    @staticmethod
    def load_data(file_path):
        data = []
        with open(file_path, encoding='utf-8') as f:
            raw_data = f.readlines()
            for dt in tqdm(raw_data):
                dt = dt.rstrip('\n').split()
                label = dt[0]
                query = dt[1]
                data.append((query, label))
        logger.info(f"filepath:{file_path} read finished!")
        return data


class Collate:
    """
    拼接batch
    """
    def __init__(self, tokenizer, max_len, label2id):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label2id = label2id

    def collate_fn(self, batch):
        batch_labels = []
        batch_input_ids = []
        batch_attention_mask = []
        batch_token_type_ids = []

        for idx, (text, label) in enumerate(batch):
            output = self.tokenizer.encode_plus(text=text,
                                                max_length=self.max_len,
                                                padding='max_length',
                                                truncation='longest_first',
                                                return_attention_mask=True,
                                                return_token_type_ids=True)

            batch_input_ids.append(output['input_ids'])
            batch_attention_mask.append(output['attention_mask'])
            batch_token_type_ids.append(output['token_type_ids'])
            batch_labels.append(self.label2id[label])

        batch_input_ids = torch.tensor(batch_input_ids, dtype=torch.long)
        batch_attention_mask = torch.tensor(batch_attention_mask, dtype=torch.long)
        batch_token_type_ids = torch.tensor(batch_token_type_ids, dtype=torch.long)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)

        return {
            'input_ids': batch_input_ids,
            'attention_mask': batch_attention_mask,
            'token_type_ids': batch_token_type_ids,
            'labels': batch_labels
        }









