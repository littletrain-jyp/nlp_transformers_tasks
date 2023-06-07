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
from dataset.dataset import AiwinDataset, Collate, CnewsDataset, Baidu2020Dataset
from classification_trainer import ClassificationTrainer, classification_predict
from models.pretrained4classification import SequenceClassificationModel
from utils.CommonUtils import set_logger, set_seed, get_latest_checkpoints_dir, load_ckp

logger = logging.getLogger(__name__)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2023, help="random seed")
    parser.add_argument("--do_train", action='store_true', help='train mode')
    parser.add_argument("--do_eval", action='store_true', help='eval mode')
    parser.add_argument("--do_predict", action='store_true', help='predict mode')
    parser.add_argument("--retrain", action='store_true', help='whether to continue training')
    parser.add_argument("--load_ckpt_dir", type=str, default=None, help='continue training or inference checkpoints dir')
    parser.add_argument("--output_dir", type=str, required=True, help='output_dir')
    parser.add_argument("--output_ckpt_dir", type=str, required=True, help='output ckpt dir')
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
    logger.info(f'label2id:{label2id}')

    tokenizer = AutoTokenizer.from_pretrained(args.model.pretrained_name_or_path)
    Collator = Collate(tokenizer, args.train.max_length, label2id, args.dataset.problem_type)


    train_dataset = CnewsDataset(file_path=args.dataset.train_dataset, has_label=True,
                                 label2id=label2id, id2label=id2label)
    # train_dataset = Baidu2020Dataset(file_path=args.dataset.train_dataset, has_label=True,
    #                              label2id=label2id, id2label=id2label)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.train.per_device_batch_size,
                              collate_fn=Collator.collate_fn,
                              shuffle=True)

    dev_dataset = CnewsDataset(file_path=args.dataset.eval_dataset, has_label=True,
                                label2id=label2id, id2label=id2label)
    # dev_dataset = Baidu2020Dataset(file_path=args.dataset.eval_dataset, has_label=True,
    #                             label2id=label2id, id2label=id2label)
    dev_loader = DataLoader(dataset=dev_dataset,
                             batch_size=args.train.per_device_batch_size,
                             collate_fn=Collator.collate_fn,
                             shuffle=False)

    test_dataset = CnewsDataset(file_path=args.dataset.test_dataset, has_label=True,
                                label2id=label2id, id2label=id2label)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.train.per_device_batch_size,
                             collate_fn=Collator.collate_fn,
                             shuffle=False)

    # predict_dataset = Baidu2020Dataset(file_path=args.dataset.predict_dataset, has_label=False,
    #                             label2id=label2id, id2label=id2label)
    # predict_loader = DataLoader(dataset=predict_dataset,
    #                          batch_size=args.train.per_device_batch_size,
    #                          collate_fn=Collator.collate_fn,
    #                          shuffle=False)

    # --------------load model optimizer -------------
    model = SequenceClassificationModel(args.model.pretrained_name_or_path,
                                        num_labels=len(label2id),
                                        label2id=label2id,
                                        id2label=id2label,
                                        return_dict=True).model


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
    warm_steps = int(args.train.warmup_ratio * max_train_steps) # 热身步数
    lr_scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=warm_steps,
        num_training_steps=max_train_steps,
    )

    # 若继续训练或者仅评估预测，需要load_ckpt
    if args.retrain or not args.do_train:
        if not args.load_ckpt_dir:
            checkpoint_dir = get_latest_checkpoints_dir(args.output_ckpt_dir)
            if not checkpoint_dir:
                logger.info("no continue training checkpoints dir!")
                assert AssertionError ("no continue training checkpoints dir！")
        else:
            checkpoint_dir = Path(args.load_ckpt_dir)

        model, optimizer, epoch, loss, global_step = load_ckp(model, optimizer, checkpoint_dir.joinpath('best.pt'), device)
        # TODO： 目前仅加载状态，继续训练 还是会按照config的epoch 从头开始完成
        logger.info(f"---->加载checkpoint模型 ckpt_dir:{checkpoint_dir.absolute()}\n \t epoch:{epoch}, global_step:{global_step}, loss:{loss}")

    # --------------load trainer -------------
    trainer = ClassificationTrainer(args=args,
                                    device=device,
                                    tokenizer=tokenizer,
                                    model=model,
                                    optimizer=optimizer,
                                    lr_scheduler=lr_scheduler,
                                    train_loader=train_loader,
                                    dev_loader=dev_loader,
                                    test_loader=test_loader,
                                    # predict_loader=predict_loader
                                    )

    if args.do_train or args.retrain:
        # 训练完成后会自动替换为效果最好的模型，并进行验证
        trainer.train()

    if args.do_eval:
        trainer.test()

    if args.do_predict:
        # 批量预测，将其转化为predict_loader 传入trainer即可
        predict_res = trainer.predict_dataset()

        print([[id2label[ii] for ii in i] for i in predict_res[:32]])

        # 单条文本预测
        predict_res = trainer.predict(text="沉睡魔咒2》正在中国大银幕热映。该片开画（10月18日）当天不及另外两部新片《双子杀手》《航海王：狂热行动》，不过很快在次日实现反超，并蝉联了5天的单日票房冠军。")

        print(id2label[predict_res[0]])

    print('finish')












