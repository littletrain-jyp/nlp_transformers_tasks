# -*- coding: utf-8 -*-
# @Time         : 2023/4/18 17:22
# @Author       : Yupeng Ji
# @File         : pretrained4classification.py
# @Description  :


import logging
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig,AutoModel, BertModel, BertForSequenceClassification

logger = logging.getLogger(__name__)

# 序列分类
class SequenceClassificationModel:
    def __init__(self, pretrained_name_or_path, **kwargs):
        try:
            self.config = AutoConfig.from_pretrained(pretrained_name_or_path, **kwargs)
            # 如果用from_config加载不会加载模型权重，只影响模型配置，所以要用from_pretrained
            self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_name_or_path,
                                                                            config=self.config)
        except Exception as e:
            logger.error(f"{e}")
            assert NotImplementedError (f"model:{pretrained_name_or_path} not support!")
