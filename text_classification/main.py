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
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from transformers import  AutoTokenizer, get_scheduler
from dataset.dataset import AiwinDataset, Collate, CnewsDataset
from classification_trainer import ClassificationTrainer, classification_predict
from models.pretrained4classification import SequenceClassificationModel
from utils.CommonUtils import set_logger, set_seed, get_latest_checkpoints_dir

logger = logging.getLogger(__name__)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2023, help="random seed")
    parser.add_argument("--do_train", action='store_true', help='train mode')
    parser.add_argument("--do_eval", action='store_true', help='eval mode')
    parser.add_argument("--do_predict", action='store_true', help='predict mode')
    parser.add_argument("--retrain", action='store_true', help='whether to continue training')
    parser.add_argument("--retrain_ckpt_dir", type=str, default=None, help='continue training checkpoints dir')
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

    args.output_ckpt_dir = './runs/test/'
    set_seed(args.seed)
    set_logger(args.output_log_dir)

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

    logger.info(f"***** args *****\n{args}\n")
    # --------------load data-------------
    # load label
    with open(args.dataset.label_dir, 'r') as f:
        labels = f.readlines()
    labels = [label.strip() for label in labels]
    label2id = dict(zip(labels, list(range(len(labels)))))
    id2label = dict(zip(list(range(len(labels))), labels))

    tokenizer = AutoTokenizer.from_pretrained(args.model.pretrained_name_or_path)
    Collator = Collate(tokenizer, args.train.max_length, label2id)

    train_dataset = CnewsDataset(file_path=args.dataset.train_dataset, has_label=True,
                                 label2id=label2id, id2label=id2label)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.train.per_device_batch_size,
                              collate_fn=Collator.collate_fn,
                              shuffle=True)

    eval_dataset = CnewsDataset(file_path=args.dataset.eval_dataset, has_label=True,
                                label2id=label2id, id2label=id2label)
    eval_loader = DataLoader(dataset=eval_dataset,
                             batch_size=args.train.per_device_batch_size,
                             collate_fn=Collator.collate_fn,
                             shuffle=False)

    test_dataset = CnewsDataset(file_path=args.dataset.test_dataset, has_label=True,
                                label2id=label2id, id2label=id2label)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.train.per_device_batch_size,
                             collate_fn=Collator.collate_fn,
                             shuffle=False)

    # --------------load model-------------

    model = SequenceClassificationModel(args.model.pretrained_name_or_path,
                                        num_labels=10,
                                        label2id=label2id,
                                        id2label=id2label,
                                        classifier_dropout=0.3,
                                        return_dict=True).model

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
        # optimizer = torch.optim.Adam(params=model.parameters(), lr=args.train.learning_rate)

        # 设置lr_scheduler
        num_update_steps_per_epoch = len(train_loader) # 每个epoch 步数
        max_train_steps = args.train.train_epochs * num_update_steps_per_epoch # 最大训练步数
        warm_steps = int(args.train.warmup_ratio * max_train_steps) # 热身步数
        lr_scheduler = get_scheduler(
            name='linear',
            optimizer=optimizer,
            num_warmup_steps=warm_steps,
            num_training_steps=max_train_steps,
        )

        # # 余弦退火曲线
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_train_steps)


        if args.retrain:
            if not args.retrain_ckpt_dir:
                checkpoint_dir = get_latest_checkpoints_dir(args.output_ckpt_dir)
                if not checkpoint_dir:
                    logger.info("no continue training checkpoints dir!")
                    assert AssertionError ("no continue training checkpoints dir！")
            else:
                checkpoint_dir = Path(args.retrain_ckpt_dir)

            ckpt = torch.load(checkpoint_dir.joinpath('best.pt'))
            model.load_state_dict(ckpt['state_dict'])
            optimizer.load_state_dict(ckpt['optimizer'])

            logger.info(f"---->加载模型继续训练，ckpt_dir:{checkpoint_dir.absolute()}\n \t epoch:{ckpt['epoch']}, global_step:{ckpt['global_step']}, loss:{ckpt['loss']}")




    trainer = ClassificationTrainer(args=args,
                                    device=device,
                                    model=model,
                                    optimizer=optimizer,
                                    lr_scheduler=lr_scheduler,
                                    train_loader=train_loader,
                                    dev_loader=test_loader,
                                    test_loader=test_loader)

    if args.do_train:
        trainer.train()

    if args.do_eval:
        trainer.dev()
        trainer.test()


    print('finish')












