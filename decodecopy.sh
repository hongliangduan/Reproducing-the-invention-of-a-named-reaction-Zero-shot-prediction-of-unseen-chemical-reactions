#!/usr/bin/env bash

PROBLEM=translate_retro_syn
MODEL=transformer
HPARAMS=transformer_base_single_gpu
DATA_DIR=./t2t_data
TMP_DIR=./tmp/t2t_datagen
TRAIN_DIR=./t2t_train/$PROBLEM/$MODEL-$HPARAMS

DECODE_FILE=$DATA_DIR/decode_this.txt

BEAM_SIZE=4
ALPHA=0.6

mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR

t2t-decoder \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" \
  --decode_from_file=$DECODE_FILE \
  --decode_to_file=200203_1.chem\
  --hparams='batch_size=6144, hidden_size=256, layer_prepostprocess_dropout=0.3'
