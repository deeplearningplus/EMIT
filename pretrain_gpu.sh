#!/bin/bash

python src/pretrain.py \
--train-file data/pretrained.trn.txt.gz \
--val-file data/pretrained.val.txt.gz \
--max-length 130 \
--epochs 40 -b 32 --print-freq 10 --workers 2 \
--wd 0.01 \
--device "cuda:0" \
--json-file model/config.json \
--tokenizer-dir model \
--output-dir model-example

