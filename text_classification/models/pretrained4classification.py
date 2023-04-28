# -*- coding: utf-8 -*-
# @Time         : 2023/4/18 17:22
# @Author       : Yupeng Ji
# @File         : pretrained4classification.py
# @Description  :


import logging
from transformers import AutoModelForSequenceClassification, AutoConfig

logger = logging.getLogger(__name__)

# 序列分类
class SequenceClassificationModel:
    def __init__(self, pretrained_name_or_path, **kwargs):
        try:
            self.config = AutoConfig.from_pretrained(pretrained_name_or_path, **kwargs)
            self.model = AutoModelForSequenceClassification.from_config(self.config)
        except:
            assert NotImplementedError (f"model:{pretrained_name_or_path} not support!")







