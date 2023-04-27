#!/bin/bash

python src/linear_probe.py \
--pretrained_path model \
--train_file data/targeted_BS_HCC_train_fold0.csv.gz \
--test_file data/targeted_BS_HCC_test_fold0.csv.gz

