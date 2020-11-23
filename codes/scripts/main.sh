#!/usr/bin/env bash

python main.py \
 --train \
 --val \
 --test \
 --gpu 0\
 --scheduler ReduceLROnPlateau \
 --datasplit tr0.1_dev0.1_t0.8 \
 --aug_p 0.2 0.4 0.2 0.2\
 --num_workers 2  \
 --batch_size 4 \
 --model ours \
 --image_size 512 512 \
 --loss_weights 0.01 0.01 1.0\
 --use_hint\
 --enchint\
 --version exp1\
 --opt_unit_dist_unit_vector_size 0.02