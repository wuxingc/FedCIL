#!/bin/bash

  python main.py \
  --dataset tiny_imagenet \
  --method bbb \
  --tasks 20 \
  --num_users 5 \
  --com_round 100 \
  --local_ep 2 \
  --beta 1.0 \
  --gpu 5 \
  --seed 2023 \
  --mem_size 500 \
  --w_old 3.0 \
  --w_new 1.0 \
  --tau_old 0.9 \
  --tau_new 1.1 \
  --local_lr 0.05 \
  --weight_decay 1e-5 \
  --local_bs 128 \
  --lambda_bmr 0.1 \
  --bmr_margin 0.5
