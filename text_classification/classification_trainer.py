# -*- coding: utf-8 -*-
# @Time         : 2023/4/28 15:49
# @Author       : Yupeng Ji
# @File         : classification_trainer.py
# @Description  :

import os
import torch
import logging
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from typing import List
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoConfig
from sklearn.metrics import accuracy_score, f1_score, classification_report

logger = logging.getLogger(__name__)

# 训练器
class ClassificationTrainer:
    def __init__(self, args, device, model, optimizer, lr_scheduler,
                 train_loader, dev_loader, test_loader=None, predict_loader=None,):
        self.args = args
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.predict_loader = predict_loader
        self.model.to(self.device)

    def load_ckp(self, model, optimizer, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return model, optimizer, epoch, loss

    def load_model(self, model, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        logger.info(f"---{checkpoint_path} read successfully! ")
        return model

    def save_ckp(self, state, checkpoint_path):
        torch.save(state, checkpoint_path)
        logger.info(f"---{checkpoint_path} saved successfully! ")

    def train(self):
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_loader.dataset) * self.args.train.train_epochs}")
        logger.info(f"  Num Epochs = {self.args.train.train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.args.train.per_device_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.args.train.gradient_accumulation_steps}")

        total_step = len(self.train_loader) * self.args.train.train_epochs
        global_step = 0
        best_select_model_metric = 0.0
        for epoch in range(self.args.train.train_epochs):
            for train_step, train_data in enumerate(tqdm(self.train_loader)):
                self.model.train()

                input_ids = train_data['input_ids'].to(self.device)
                attention_masks = train_data['attention_masks'].to(self.device)
                token_type_ids = train_data['token_type_ids'].to(self.device)
                labels = train_data['labels'].to(self.device)

                train_outputs = self.model(input_ids, attention_masks, token_type_ids, labels)
                loss = train_outputs.loss / self.args.train.gradient_accumulation_steps # 损失标准化
                loss.backward() # 反向传播，计算梯度

                if (train_step + 1) % self.args.train.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm(self.model.parameters(), self.args.train.max_grad_norm)
                    self.optimizer.step()  #优化器更新参数
                    self.lr_scheduler.step() # lr_scheduler 更新参数
                    self.optimizer.zero_grad() # 优化器的梯度清零，注意区分model.zero_grad

                if global_step % self.args.train.logging_steps == 0:
                    logger.info(
                        f"【train】 epoch：{epoch} epoch_step:{train_step}/{len(self.train_loader)}, "
                        f"global_step:{global_step}/{total_step} loss：{loss.item():.6f}")
                global_step += 1
                # 评估并保存checkpoint
                if global_step % self.args.train.eval_steps == 0:
                    dev_loss, dev_outputs, dev_targets = self.dev()
                    metrics = self.get_metrics(dev_outputs, dev_targets)
                    try:
                        select_model_metric = metrics[self.args.train.select_model_metric]
                    except:
                        raise AssertionError (f"{self.args.train.select_model_metric} not in model metrics !")

                    logger.info(
                        f"【dev】 loss：{dev_loss:.6f} {self.print_metrics(metrics)}")
                    if select_model_metric > best_select_model_metric:

                        checkpoint = {
                            'epoch': epoch,
                            'loss': dev_loss,
                            'state_dict': self.model.state_dict(),
                            'opt·imizer': self.optimizer.state_dict(),
                        }
                        best_select_model_metric = select_model_metric
                        checkpoint_dir_name = f"checkpoint_epoch{epoch}_step{global_step}"
                        save_path = os.path.join(self.args.output_ckpt_dir, checkpoint_dir_name)
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        checkpoint_path = os.path.join(save_path, 'best.pt')
                        logger.info(f"------------>保存当前最好的模型到: {checkpoint_path}")
                        self.save_ckp(checkpoint, checkpoint_path)

    def dev(self):
        total_loss, dev_outputs, dev_targets = self._eval_loop(self.dev_loader, "eval")
        metrics = self.get_metrics(dev_outputs, dev_targets)
        logger.info(f"--eval result: {self.print_metrics(metrics)}")

    def test(self):
        total_loss, test_outputs, test_targets = self._eval_loop(self.test_loader, "test")
        metrics = self.get_metrics(test_outputs, test_targets)
        logger.info(f"--test result: {self.print_metrics(metrics)}")

    def predict_dataset(self):
        return self._predict_loop(self.predict_loader, 'predict')

    def _eval_loop(self,
                   dataloader: DataLoader,
                   description: str, ):
        logger.info(f"***** Running {description} *****")
        logger.info(f"  Num examples = {len(dataloader)}")
        self.model.eval()
        total_loss = 0.0
        dev_outputs = []
        dev_targets = []
        with torch.no_grad():
            for dev_step, dev_data in enumerate(tqdm(dataloader)):
                input_ids = dev_data['input_ids'].to(self.device)
                attention_masks = dev_data['attention_masks'].to(self.device)
                token_type_ids = dev_data['token_type_ids'].to(self.device)
                labels = dev_data['labels'].to(self.device)

                outputs = self.model(input_ids, attention_masks, token_type_ids, labels)
                # val_loss = val_loss + ((1 / (dev_step + 1))) * (loss.item() - val_loss)
                total_loss += outputs.loss.item()
                outputs_res = np.argmax(outputs.logits, axis=1).flatten()
                dev_outputs.extend(outputs_res.tolist())
                dev_targets.extend(labels.tolist())

        return total_loss, dev_outputs, dev_targets

    def _predict_loop(self,
                      dataloader: DataLoader,
                      description: str):
        logger.info(f"***** Running {description} *****")
        logger.info(f"  Num examples = {len(dataloader)}")
        self.model.eval()
        dev_outputs = []
        with torch.no_grad():
            for step, batch_data in enumerate(tqdm(dataloader)):
                input_ids = batch_data['input_ids'].to(self.device)
                attention_masks = batch_data['attention_masks'].to(self.device)
                token_type_ids = batch_data['token_type_ids'].to(self.device)

                outputs = self.model(input_ids, attention_masks, token_type_ids)
                outputs_res = np.argmax(outputs.logits, axis=1).flatten()
                dev_outputs.extend(outputs_res.tolist())

        return dev_outputs

    def get_class_weight(self, dataset):
        """
        获取每个类的权重，用于出现类别imbalance 情况时进行少样本类别系数补偿
        :param dataset:
        :return:
        """
        class_count_dict = {}
        for data in dataset:
            label = data.label
            if label not in class_count_dict:
                class_count_dict[label] = 1
            else:
                class_count_dict[label] += 1
        class_count_dict = dict(sorted(class_count_dict.items(), key=lambda x: x[0]))
        print('[+] sample(s) count of each class: ', class_count_dict)

        class_weight_dict = {}
        class_count_sorted_tuple = sorted(class_count_dict.items(), key=lambda x: x[1], reverse=True)
        for i in range(len(class_count_sorted_tuple)):
            if i == 0:
                class_weight_dict[class_count_sorted_tuple[0][0]] = 1.0  # 数量最多的类别权重为1.0
            else:
                scale_ratio = class_count_sorted_tuple[0][1] / class_count_sorted_tuple[i][1]
                scale_ratio = min(scale_ratio, self.args.train.class_weight_max_scale_ratio)  # 数量少多少倍，loss就scale几倍
                class_weight_dict[class_count_sorted_tuple[i][0]] = scale_ratio
        print('[+] weight(s) of each class: ', class_weight_dict)
        return class_weight_dict

    def get_metrics(self, outputs, targets) -> dict:
        accuracy = accuracy_score(targets, outputs)
        micro_f1 = f1_score(targets, outputs, average='micro')
        macro_f1 = f1_score(targets, outputs, average='macro')
        return {
            "accuracy": accuracy,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1
        }

    def print_metrics(self, metrics: dict) -> str:
        metrics_str = ''
        for k,v in metrics:
            metrics_str += f'{k}:{v:.6f}'
        return metrics_str

    def get_classification_report(self, outputs, targets, labels):
        report = classification_report(targets, outputs, target_names=labels)
        return report


def classification_predict(tokenizer, text, device, args, model):
    if isinstance(text, List):
        assert len(text) <= args.train.per_device_batch_size ("please use predict_loader to predict")
    model.eval()
    model.to(device)
    with torch.no_grad():
        inputs = tokenizer.encode_plus(text=text,
                                       add_special_tokens=True,
                                       max_length=args.train.max_length,
                                       truncation='longest_first',
                                       padding="max_length",
                                       return_token_type_ids=True,
                                       return_attention_mask=True,
                                       return_tensors='pt')
        token_ids = inputs['input_ids'].to(device)
        attention_masks = inputs['attention_mask'].to(device)
        token_type_ids = inputs['token_type_ids'].to(device)

        outputs = model(token_ids, attention_masks, token_type_ids)
        outputs_res = np.argmax(outputs.logits, axis=1).flatten().tolist()
        if len(outputs_res) != 0:
            return outputs_res
        else:
            return '不好意思，我没有识别出来'

