# -*- coding: utf-8 -*-
# @Time         : 2023/4/19 14:27
# @Author       : Yupeng Ji
# @File         : main.py
# @Description  :

import os
import logging
import yaml
import argparse

from models.pretrained4classification import SequenceClassificationModel

logger = logging.getLogger(__name__)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2023, help="random seed")
    parser.add_argument("--do_train", action='store_true', help='train mode')
    parser.add_argument("--do_eval", action='store_true', help='eval mode')
    parser.add_argument("--do_test", action='store_true', help='test mode')
    parser.add_argument("--config", type=str, required=True, help='YAML config file path')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        conf = yaml.safe_load(f)

    args = vars(args)
    args.update(conf)

    args['model'] = argparse.Namespace(**args['model'])

    # --------------load data-------------



    # --------------load model-------------







