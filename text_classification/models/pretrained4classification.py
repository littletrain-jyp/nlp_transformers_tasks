# -*- coding: utf-8 -*-
# @Time         : 2023/4/18 17:22
# @Author       : Yupeng Ji
# @File         : pretrained4classification.py
# @Description  :

from transformers import AutoModelForSequenceClassification, AutoConfig
import torch.nn as nn

# 序列分类
class SequenceClassificationModel(nn.Module):
    def __init__(self, pretrained_name_or_path, num_labels):
        super().__init__()
        try:
            self.config = AutoConfig.from_pretrained(pretrained_name_or_path)
            self.config.num_labels = num_labels
            self.model = AutoModelForSequenceClassification.from_config(self.config)
        except:
            assert NotImplementedError (f"model:{pretrained_name_or_path} not support!")









