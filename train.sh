#!/usr/bin/env bash

PROBLEM=translate_retro_syn
MODEL=transformer
HPARAMS=transformer_base_single_gpu
DATA_DIR=./t2t_data
TMP_DIR=./tmp/t2t_datagen
TRAIN_DIR=./t2t_train/$PROBLEM/$MODEL-$HPARAMS

mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR

t2t-trainer \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --train_steps=2000000 \
  --hparams='batch_size=6144, hidden_size=256, layer_prepostprocess_dropout=0.3' \
  --sh datagen.sh=60




