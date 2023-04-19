# -*- coding: utf-8 -*-
# @Time         : 2023/4/18 17:22
# @Author       : Yupeng Ji
# @File         : pretrained4classification.py
# @Description  :

from transformers import AutoModelForSequenceClassification, AutoConfig
import torch.nn as nn

# 序列分类
class SequenceClassificationModel(nn.Module):
    def __int__(self, args):
        super(SequenceClassificationModel, self).__int__()
        try:
            self.config = AutoConfig.from_pretrained(args.model.pretrained_name_or_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.config)
        except:
            assert NotImplementedError (f"model:{args.model.pretrained_name_or_path} not support!")








