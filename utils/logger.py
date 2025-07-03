#!/usr/bin/python
# -*- coding:utf-8 -*-

import logging


def setlogger(log_file):
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
