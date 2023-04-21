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
from  dataclasses import dataclass
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class InputExample:
    text: str
    label: int = None

class ClassificationDataset(Dataset):
    """
    基类，直接传入文件名或者list形式的数据都可以
    """
    def __init__(self, file_path=None, data=None, **kwargs):
        self.kwargs = kwargs
        if isinstance(file_path, (str, list)):
            self.data = self.load_data(file_path, kwargs['has_label'])
        elif isinstance(data, list):
            self.data = data
        else:
            raise ValueError ("The input args shall be str format file_path / list format dataset")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def _process_sample(self, sample_dt, has_label=True):
        raise ImportError ("basecalss not support yet! You should overwrite it!")

    def load_data(self, file_path, has_label=True):
        raise ImportError ("basecalss not support yet! You should overwrite it!")


# 自定义处理相关数据
class CnewsDataset(ClassificationDataset):
    def _process_sample(self, sample_dt, has_label=True):
        dt = sample_dt.split()
        if has_label:
            return InputExample(text=dt[1], label=dt[0])
        else:
            return InputExample(text=dt[0])

    def load_data(self, file_path, has_label=True):
        data = []
        with open(file_path, encoding='utf-8') as f:
            raw_data = f.readlines()
            for dt in tqdm(raw_data):
                data.append(self._process_sample(dt.rstrip('\n')), has_label)
        logger.info(f"filepath:{file_path} read finished!")
        return data

class AiwinDataset(ClassificationDataset):
    def _process_sample(self, sample_dt, has_label=True):
        label = sample_dt['label']
        text = f"header:{sample_dt['header']}\ntitle:{sample_dt['title']}\nparagraph:{sample_dt['paragraph']}\nfooter:{sample_dt['footer']}"
        if has_label:
            return InputExample(text=text, label=label)
        else:
            return InputExample(text=text)

    def load_data(self, file_path, has_label=True):
        data = []
        with open(file_path, encoding='utf-8') as f:
            raw_data = f.readlines()
            for idx, dt in enumerate(tqdm(raw_data)):
                dt = self._process_sample(eval(dt.rstrip('\n')), has_label)
                data.append(dt)
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

        for idx, data in enumerate(batch):
            text = data.text
            label = data.label

            output = self.tokenizer.encode_plus(text=text,
                                                max_length=self.max_len,
                                                padding='max_length',
                                                truncation='longest_first',
                                                return_attention_mask=True,
                                                return_token_type_ids=True)

            batch_input_ids.append(output['input_ids'])
            batch_attention_mask.append(output['attention_mask'])
            batch_token_type_ids.append(output['token_type_ids'])
            if label:
                batch_labels.append(label)

        batch_input_ids = torch.tensor(batch_input_ids, dtype=torch.long)
        batch_attention_mask = torch.tensor(batch_attention_mask, dtype=torch.long)
        batch_token_type_ids = torch.tensor(batch_token_type_ids, dtype=torch.long)
        input_features = {
            'input_ids': batch_input_ids,
            'attention_mask': batch_attention_mask,
            'token_type_ids': batch_token_type_ids
        }

        if len(batch_labels) != 0:
            batch_labels = torch.tensor(batch_labels, dtype=torch.long)
            input_features.update({'labels': batch_labels})

        return input_features


if __name__ == "__main__":
    # cnews_data = CnewsDataset('../data/cnews/cnews.test.txt')
    aiwin_data = AiwinDataset(file_path='../data/aiwin/train_dataset.txt', has_label=False)
    print(len(aiwin_data))








