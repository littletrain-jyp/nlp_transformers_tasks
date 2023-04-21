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
from transformers import TrainingArguments, Trainer, AutoTokenizer
from torch.utils.data import DataLoader

from dataset.dataset import AiwinDataset, Collate
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
    labels2id = [{}.update({v: k}) for k, v in enumerate(labels)]
    id2labels = [{}.update({k: v}) for k, v in enumerate(labels)]

    train_dataset = AiwinDataset(file_path=args.dataset.eval_dataset, has_label=True)
    train_loader = DataLoader(dataset=train_dataset, shuffle=True)

    # --------------load model-------------
    tokenizer = AutoTokenizer.from_pretrained(args.model.pretrained_name_or_path)
    model = SequenceClassificationModel(args.model.pretrained_name_or_path, 10)
    Collator = Collate(tokenizer, args.train.max_length, labels2id)
    # 配置Training TrainingArguments
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        logging_dir=args.output_log_dir,
        do_train=args.do_train,
        do_eval=args.do_eval,
        do_predict=args.do_predict,
        evaluation_strategy='steps',
        eval_steps=args.train.steps_per_epoch,
        prediction_loss_only=True,
        per_device_train_batch_size=args.train.per_device_batch_size,
        per_device_eval_batch_size=args.train.per_device_batch_size,
        gradient_accumulation_steps=args.train.gradient_accumulation_steps,
        num_train_epochs=args.train.num_epoch,
        learning_rate= args.train.learning_rate,
        save_steps=args.train.steps_per_epoch,
        save_strategy='steps',
        save_total_limit=3,
        load_best_model_at_end=True,
    )

    trainer = Trainer(model=model.model,
                      args=train_args,
                      data_collator=Collator.collate_fn,
                      train_dataset=train_dataset)

    if args.do_train:
        trainer.train()

    print('finish')












