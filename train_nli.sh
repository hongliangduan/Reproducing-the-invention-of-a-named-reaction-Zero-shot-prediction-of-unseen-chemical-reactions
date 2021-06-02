#!/usr/bin/env bash

PROBLEM=question_nli
MODEL=transformer
HPARAMS=transformer_tiny
DATA_DIR=./t2t_data
TMP_DIR=./tmp/t2t_datagen
TRAIN_DIR=./t2t_train/$PROBLEM

mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR

t2t-trainer \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --train_steps=2000000 \
  --save_ckpt_steps=10000





