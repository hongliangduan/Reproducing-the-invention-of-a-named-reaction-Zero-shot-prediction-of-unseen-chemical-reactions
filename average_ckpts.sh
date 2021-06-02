#!/usr/bin/env bash




python -m utils.avg_checkpoints \
    --num_last_checkpoints=20 \
    --checkpoints ='model.ckpt-864000, model.ckpt-865000, model.ckpt-866000, model.ckpt-867000, model.ckpt-868000,
    model.ckpt-869000, model.ckpt-870000, model.ckpt-871000, model.ckpt-872000, model.ckpt-873000,
    model.ckpt-874000, model.ckpt-875000, model.ckpt-876000, model.ckpt-877000, model.ckpt-878000,
    model.ckpt-879000, model.ckpt-880000, model.ckpt-881000, model.ckpt-882000, model.ckpt-883000' \
    --prefix="/home/ubuntu/t2t_train/translate_enzh_wmt32k/transformer-transformer_base_single_gpu/Retro_syn/Batch_size_8192_2/" \
    --output_path="/home/ubuntu/t2t_train/translate_enzh_wmt32k/transformer-transformer_base_single_gpu/Retro_syn/"

