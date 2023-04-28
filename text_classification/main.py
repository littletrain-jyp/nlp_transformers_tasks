# -*- coding: utf-8 -*-
# @Time         : 2023/4/19 14:27
# @Author       : Yupeng Ji
# @File         : main.py
# @Description  :

import os
import logging
import torch
import yaml
import argparse
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from transformers import TrainingArguments, Trainer, AutoTokenizer, \
    get_scheduler, BertForSequenceClassification

from dataset.dataset import AiwinDataset, Collate
from classification_trainer import ClassificationTrainer, classification_predict
from models.pretrained4classification import SequenceClassificationModel

logger = logging.getLogger(__name__)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2023, help="random seed")
    parser.add_argument("--do_train", action='store_true', help='train mode')
    parser.add_argument("--do_eval", action='store_true', help='eval mode')
    parser.add_argument("--do_predict", action='store_true', help='predict mode')
    parser.add_argument("--output_dir", type=str, required=True, help='output_dir')
    parser.add_argument("--output_log_dir", type=str, required=True, help='output log dir')
    parser.add_argument("--config", type=str, required=True, help='YAML config file path')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        conf = yaml.safe_load(f)

    args = vars(args)
    args.update(conf)

    args['dataset'] = argparse.Namespace(**args['dataset'])
    args['model'] = argparse.Namespace(**args['model'])
    args['train'] = argparse.Namespace(**args['train'])
    args['run'] = argparse.Namespace(**args['run'])
    args = argparse.Namespace(**args)

    # 更新参数
    # 设置device
    device_ids = [int(i) for i in args.run.device_id.split(',')]
    if device_ids[0] == -1:
        device = torch.device('cpu')
    else:
        if len(device_ids) > 1:
            logger.info(f'use_mulit_gpu : {device_ids}')
            assert ImportError ("not support multi-gpu yet! ")
        else:
            device = torch.device(f"cuda:{device_ids[0]}")

    # --------------load data-------------
    # load label
    with open(args.dataset.label_dir, 'r') as f:
        labels = f.readlines()
    labels = [label.strip() for label in labels]
    label2id = dict(zip(labels, list(range(len(labels)))))
    id2label = dict(zip(list(range(len(labels))), labels))

    train_dataset = AiwinDataset(file_path=args.dataset.train_dataset, has_label=True)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size= args.train.per_device_batch_size,
                              shuffle=True)

    eval_dataset = AiwinDataset(file_path=args.dataset.eval_dataset, has_label=True)
    eval_loader = DataLoader(dataset=eval_dataset,
                             batch_size=args.train.per_device_batch_size,
                             shuffle=False)

    # --------------load model-------------
    tokenizer = AutoTokenizer.from_pretrained(args.model.pretrained_name_or_path)
    model = SequenceClassificationModel(args.model.pretrained_name_or_path,
                                        num_labels=10,
                                        label2id=label2id,
                                        id2label=id2label,
                                        return_dict=True).model
    Collator = Collate(tokenizer, args.train.max_length, id2label)

    if args.do_train:
        # 设置权重衰减参数
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.train.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            }]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.train.learning_rate)

        # 设置lr_scheduler
        num_update_steps_per_epoch = len(train_loader) # 每个epoch 步数
        max_train_steps = args.train.train_epochs * num_update_steps_per_epoch # 最大训练步数
        warm_steps = int(args.warmup_ratio * max_train_steps) # 热身步数
        # 线性
        lr_scheduler = get_scheduler(
            name='linear',
            optimizer=optimizer,
            num_warmup_steps=warm_steps,
            num_training_steps=max_train_steps,
        )
        # # 余弦退火曲线
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_train_steps)

    trainer = ClassificationTrainer(args=args,
                                    device=device,
                                    model=model,
                                    optimizer=optimizer,
                                    lr_scheduler=lr_scheduler,
                                    train_loader=train_loader,
                                    dev_loader=eval_loader)

    if args.do_train:
        trainer.train()

    if args.do_eval:
        trainer.evaluate()
        predictions, labels, loss = trainer.predict(eval_dataset)
        prediction_list = np.argmax(predictions, axis=1).flatten().tolist()
        labels_list = labels.tolist()
        print(classification_report(labels_list, prediction_list))



    print('finish')












