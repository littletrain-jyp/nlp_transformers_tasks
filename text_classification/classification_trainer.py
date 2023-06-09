# -*- coding: utf-8 -*-
# @Time         : 2023/4/28 15:49
# @Author       : Yupeng Ji
# @File         : classification_trainer.py
# @Description  :

import os
import time
import torch
import logging
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from typing import List
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoModelForSequenceClassification, AutoConfig, BertForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report

from utils.CommonUtils import keep_checkpoints_num, get_time_diff, save_ckp, load_model
from utils.adversarial import FGM

logger = logging.getLogger(__name__)

# 训练器
class ClassificationTrainer:
    def __init__(self, args, device, tokenizer, model, optimizer, lr_scheduler,
                 train_loader, dev_loader, test_loader=None, predict_loader=None,):
        self.args = args
        self.device = device
        self.tokenizer = tokenizer
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.predict_loader = predict_loader
        self.model.to(self.device)

        if self.args.dataset.problem_type == 'single_label_classification':
            self.train_class_weight = self.get_class_weight(train_loader.dataset)
        if self.args.train.with_amp:
            self.scaler = GradScaler()

        self.best_ckpt_dir = None

    def train(self):
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_loader.dataset) * self.args.train.train_epochs}")
        logger.info(f"  Num Epochs = {self.args.train.train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.args.train.per_device_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.args.train.gradient_accumulation_steps}")

        if self.args.train.use_adversarial:
            fgm = FGM(self.model, emb_name='word_embedding')

        total_step = len(self.train_loader) * self.args.train.train_epochs
        global_step = 1
        best_select_model_metric = 0.0
        start_time = time.time()
        for epoch in range(self.args.train.train_epochs):
            epoch_start_time = time.time()
            for train_step, train_data in enumerate(tqdm(self.train_loader)):
                self.model.train()

                input_ids = train_data['input_ids'].to(self.device)
                attention_mask = train_data['attention_mask'].to(self.device)
                token_type_ids = train_data['token_type_ids'].to(self.device)
                labels = train_data['labels'].to(self.device)

                if self.args.train.with_amp:
                    with autocast():
                        train_outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            labels=labels)
                        loss = train_outputs.loss
                else:
                    train_outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=labels)
                    loss = train_outputs.loss

                # 按照权重类别更新loss
                if self.args.train.use_class_weight and self.args.dataset.problem_type == 'single_label_classification':
                    labels = train_data['labels'].tolist()
                    weights = [self.train_class_weight[label] for label in labels]
                    weight = sum(weights) / len(weights)
                    loss *= weight

                loss = loss / self.args.train.gradient_accumulation_steps # 损失标准化

                if self.args.train.with_amp:
                    self.scaler.scale(loss).backward() # 半精度的数值范围有限，需要用它放大
                else:
                    loss.backward() # 反向传播，计算梯度

                if self.args.train.use_adversarial:
                    fgm.attack()
                    loss_adv = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=labels).loss
                    loss_adv.backward()
                    fgm.restore()

                if (train_step + 1) % self.args.train.gradient_accumulation_steps == 0:
                    if self.args.train.with_amp:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.train.max_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.train.max_grad_norm)
                        self.optimizer.step()  #优化器更新参数

                    self.lr_scheduler.step() # lr_scheduler 更新参数
                    self.optimizer.zero_grad() # 优化器的梯度清零，注意区分model.zero_grad

                if global_step % self.args.train.logging_steps == 0:
                    logger.info(
                        f"【train】 epoch：{epoch} epoch_step:{train_step+1}/{len(self.train_loader)}, "
                        f"global_step:{global_step}/{total_step} loss：{loss.item():.6f} "
                        f"epoch_cost_time:{get_time_diff(epoch_start_time)} total_epoch_time:{(get_time_diff(epoch_start_time) / train_step * len(self.train_loader))} "
                        f"total_cost_time:{get_time_diff(start_time)} total_remaining_time:{(get_time_diff(start_time) / global_step * total_step)} ")

                # 评估并保存checkpoint
                if global_step % self.args.train.eval_steps == 0:
                    dev_loss, dev_outputs, dev_targets, metrics = self.dev(self.dev_loader)
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
                            'global_step': global_step,
                            'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                        }
                        best_select_model_metric = select_model_metric

                        keep_checkpoints_num(self.args.output_ckpt_dir) # 删掉多余的ckpt
                        checkpoint_dir_name = f"checkpoint_epoch{epoch}_step{global_step}"
                        save_path = os.path.join(self.args.output_ckpt_dir, checkpoint_dir_name)
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        checkpoint_path = os.path.join(save_path, 'best.pt')
                        self.best_ckpt_dir = checkpoint_path
                        logger.info(f"------------>保存当前最好的模型到: {checkpoint_path}")
                        save_ckp(checkpoint, checkpoint_path)

                global_step += 1

        # 所有训练结束后，加载验证集效果最后的模型进行验证
        logger.info(f"------------>加载当前最好的模型到: {self.best_ckpt_dir}")
        self.model = load_model(self.model, self.best_ckpt_dir, self.device)
        self.dev()

    def dev(self, dev_loader=None, desc="eval", ckpt_dir=None):
        if not dev_loader:
            dev_loader = self.dev_loader
        if ckpt_dir:
            self.model = load_model(self.model, ckpt_dir, self.device)

        total_loss, dev_outputs, dev_targets = self._eval_loop(dev_loader, desc)
        metrics = self.get_metrics(dev_outputs, dev_targets)
        logger.info(f"\n{desc} result: {self.print_metrics(metrics)}")
        logger.info(self.get_classification_report(dev_outputs, dev_targets, self.train_loader.dataset.label2id.keys()))
        return total_loss, dev_outputs, dev_targets, metrics

    def test(self):
        return self.dev(dev_loader=self.test_loader, desc="test")

    def predict_dataset(self, ckpt_dir=None):
        """ 文件类预测 """
        # 加载模型
        if ckpt_dir:
            self.model = load_model(self.model, ckpt_dir, self.device)
        return self._predict_loop(self.predict_loader, 'predict')

    def predict(self, text, ckpt_dir=None):
        """ 单条文本预测 """
        # 加载模型
        if ckpt_dir:
            self.model = load_model(self.model, ckpt_dir, self.device)

        token_res = self.tokenizer.encode_plus(text=text,
                                            max_length=self.args.train.max_length,
                                            padding='max_length',
                                            truncation='longest_first',
                                            return_attention_mask=True,
                                            return_token_type_ids=True)
        input_ids = torch.tensor([token_res['input_ids']], dtype=torch.long).to(self.device)
        attention_mask = torch.tensor([token_res['attention_mask']], dtype=torch.long).to(self.device)
        token_type_ids = torch.tensor([token_res['token_type_ids']], dtype=torch.long).to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids)

        if self.args.dataset.problem_type == 'single_label_classification':
            outputs_res = np.argmax(outputs.logits.cpu().detach().numpy(), axis=1).flatten().tolist()
        else:
            outputs_res = torch.sigmoid(outputs.logits).cpu().detach().numpy().tolist()
            outputs_res = (np.array(outputs_res) > self.args.dataset.multi_label_classification_threshold).astype(int)
            outputs_res = np.where(outputs_res[0] == 1)[0].tolist()
        return outputs_res


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
            for dev_step, dev_data in enumerate(tqdm(dataloader, leave=False)):
                input_ids = dev_data['input_ids'].to(self.device)
                attention_mask = dev_data['attention_mask'].to(self.device)
                token_type_ids = dev_data['token_type_ids'].to(self.device)
                labels = dev_data['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels
                )

                total_loss += outputs.loss.item()

                if self.args.dataset.problem_type == 'single_label_classification':
                    outputs_res = np.argmax(outputs.logits.cpu(), axis=1).flatten()
                else:
                    outputs_res = torch.sigmoid(outputs.logits).cpu().detach().numpy().tolist()
                    outputs_res = (np.array(outputs_res) > self.args.dataset.multi_label_classification_threshold).astype(int)

                dev_outputs.extend(outputs_res.tolist())
                dev_targets.extend(labels.cpu().detach().numpy().tolist())

        return total_loss, dev_outputs, dev_targets

    def _predict_loop(self,
                      dataloader: DataLoader,
                      description: str):
        logger.info(f"***** Running {description} *****")
        logger.info(f"  Num examples = {len(dataloader)}")
        self.model.eval()
        pred_outputs = []
        with torch.no_grad():
            for step, batch_data in enumerate(tqdm(dataloader, leave=False)):
                input_ids = batch_data['input_ids'].to(self.device)
                attention_mask = batch_data['attention_mask'].to(self.device)
                token_type_ids = batch_data['token_type_ids'].to(self.device)

                outputs = self.model(input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids)

                if self.args.dataset.problem_type == 'single_label_classification':
                    outputs_res = np.argmax(outputs.logits.cpu().detach().numpy(), axis=1).flatten()
                else:
                    outputs_res = torch.sigmoid(outputs.logits).cpu().detach().numpy().tolist()
                    outputs_res = (np.array(outputs_res) > self.args.dataset.multi_label_classification_threshold).astype(int)
                    outputs_res = [np.where(i == 1)[0].tolist() for i in outputs_res]

                pred_outputs.extend(outputs_res)

        return pred_outputs

    def get_class_weight(self, dataset):
        """
        获取每个类的权重，用于出现类别imbalance 情况时进行少样本类别系数补偿, 仅支持单标签
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
        logger.info(f'[+] sample(s) count of each class: {class_count_dict}')

        class_weight_dict = {}
        class_count_sorted_tuple = sorted(class_count_dict.items(), key=lambda x: x[1], reverse=True)
        for i in range(len(class_count_sorted_tuple)):
            if i == 0:
                class_weight_dict[class_count_sorted_tuple[0][0]] = 1.0  # 数量最多的类别权重为1.0
            else:
                scale_ratio = class_count_sorted_tuple[0][1] / class_count_sorted_tuple[i][1]
                scale_ratio = min(scale_ratio, self.args.train.class_weight_max_scale_ratio)  # 数量少多少倍，loss就scale几倍
                class_weight_dict[class_count_sorted_tuple[i][0]] = scale_ratio
        logger.info(f'[+] weight(s) of each class: {class_weight_dict}')
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
        for k, v in metrics.items():
            metrics_str += f' {k}:{v:.6f} '
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
        attention_mask = inputs['attention_mask'].to(device)
        token_type_ids = inputs['token_type_ids'].to(device)

        outputs = model(token_ids, attention_mask, token_type_ids)
        outputs_res = np.argmax(outputs.logits.cpu(), axis=1).flatten().tolist()
        if len(outputs_res) != 0:
            return outputs_res
        else:
            return '不好意思，我没有识别出来'








