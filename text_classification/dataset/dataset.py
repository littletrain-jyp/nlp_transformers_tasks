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
from typing import List
from dataclasses import dataclass
from torch.utils.data import Dataset
from utils.CommonUtils import get_multi_onehot_label

logger = logging.getLogger(__name__)


@dataclass
class InputExample:
    text: str
    label: int = None

@dataclass
class MultilabelInputExample:
    text: str
    label: list[int] = None


class ClassificationDataset(Dataset):
    """
    基类，直接传入文件名或者list形式的数据都可以
    """
    def __init__(self, file_path=None, data=None, label2id=None, id2label=None, **kwargs):
        super(ClassificationDataset).__init__()
        self.label2id = label2id
        self.id2label = id2label
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
        """
        自定义处理数据, 将单标签处理为int，多标签的处理为list[int]
        :param sample_dt:
        :param has_label:
        :return:
        """
        raise ImportError ("basecalss not support yet! You should overwrite it!")


    def load_data(self, file_path, has_label=True):
        raise ImportError ("basecalss not support yet! You should overwrite it!")

class Collate:
    """
    拼接batch, 单标签的label 为 [1,batchsize], 多标签label为[batchsize, class_num]
    """
    def __init__(self, tokenizer, max_len, label2id, problem_type='single_label_classification'):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label2id = label2id
        self.problem_type = problem_type

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


            if label is not None:
                if isinstance(label, int) and self.problem_type == 'single_label_classification':
                    batch_labels.append(label)
                elif isinstance(label, List) and self.problem_type == 'multi_label_classification':
                    batch_labels.append(label)
                else:
                    logger.info(f'skip error sample: idx:{idx}, data:{data}')
                    continue

            batch_input_ids.append(output['input_ids'])
            batch_attention_mask.append(output['attention_mask'])
            batch_token_type_ids.append(output['token_type_ids'])


        batch_input_ids = torch.tensor(batch_input_ids, dtype=torch.long)
        batch_attention_mask = torch.tensor(batch_attention_mask, dtype=torch.long)
        batch_token_type_ids = torch.tensor(batch_token_type_ids, dtype=torch.long)
        input_features = {
            'input_ids': batch_input_ids,
            'attention_mask': batch_attention_mask,
            'token_type_ids': batch_token_type_ids
        }

        if len(batch_labels) != 0:
            if self.problem_type == 'single_label_classification':
                batch_labels = torch.tensor(batch_labels, dtype=torch.long)
            else:
                batch_labels = get_multi_onehot_label(batch_labels, len(self.label2id), torch.float)
            input_features.update({'labels': batch_labels})
            assert len(batch_labels) == len(batch_input_ids)


        return input_features



# 自定义处理相关数据
class CnewsDataset(ClassificationDataset):
    def _process_sample(self, sample_dt, has_label=True):
        dt = sample_dt.split('\t')

        if has_label:
            text = dt[1]
            label = dt[0]

            if isinstance(label, str):
                label = self.label2id[label]
            return InputExample(text=text, label=label)
        else:
            return InputExample(text=dt[0])

    def load_data(self, file_path, has_label=True):
        data = []
        with open(file_path, encoding='utf-8') as f:
            raw_data = f.readlines()
            for dt in tqdm(raw_data):
                data.append(self._process_sample(dt.rstrip('\n'), has_label))
        logger.info(f"filepath:{file_path} read finished!")
        return data

class AiwinDataset(ClassificationDataset):
    def _process_sample(self, sample_dt, has_label=True):
        label = sample_dt['label']
        text = f"header:{sample_dt['header']} [SEP] {sample_dt['title']} [SEP] {sample_dt['paragraph']} [SEP] {sample_dt['footer']}"
        if has_label:
            if isinstance(label, str):
                label = self.label2id[label]
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

class Baidu2020Dataset(ClassificationDataset):
    def _process_sample(self, sample_dt, has_label=True):
        text = sample_dt['text']
        if has_label:
            labels = []
            if len(sample_dt['event_list']) != 0:
                for tmp in sample_dt['event_list']:
                    labels.append(tmp['event_type'])
            labels = [self.label2id[i] for i in labels]
            return MultilabelInputExample(text=text, label=labels)
        else:
            return MultilabelInputExample(text=text)

    def load_data(self, file_path, has_label=True):
        data = []
        with open(file_path, encoding='utf-8') as f:
            raw_data = f.readlines()
            for idx, dt in enumerate(tqdm(raw_data)):
                dt = self._process_sample(eval(dt.rstrip('\n')), has_label)
                data.append(dt)
        logger.info(f"filepath:{file_path} read finished!")
        return data


if __name__ == "__main__":
    # cnews_data = CnewsDataset('../data/cnews/cnews.test.txt', has_label=True)
    # aiwin_data = AiwinDataset(file_path='../data/aiwin/train_dataset.txt', has_label=False)

    with open('./data/baidu_event_extra_2020/final_data/labels.txt', 'r') as f:
        labels = f.readlines()
    labels = [label.strip() for label in labels]
    label2id = dict(zip(labels, list(range(len(labels)))))
    id2label = dict(zip(list(range(len(labels))), labels))


    baidu_data = Baidu2020Dataset(file_path='./data/baidu_event_extra_2020/raw_data/dev.json',
                                  label2id=label2id,
                                  id2label=id2label,
                                  has_label=True)

    print(len(baidu_data))








