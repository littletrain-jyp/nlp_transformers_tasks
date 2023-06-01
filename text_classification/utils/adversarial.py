# -*- coding: utf-8 -*-
# @Time         : 2023/5/25 11:09
# @Author       : Yupeng Ji
# @File         : adversarial.py
# @Description  :

import torch

class FGM:
    def __init__(self, model, emb_name, epsilon=1.0):
        # emb_name 要换成你模型中embedding的参数名
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name and not name.startswith("moco_bert"):
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name and not name.startswith("moco_bert"):
                assert name in self.backup
                param.data = self.backup[name]
            self.backup = {}




