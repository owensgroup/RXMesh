#!/bin/bash

# Connect to a GPU node with specified resources
srun --partition gpunodes \
     -c 4 \
     --gres=gpu:rtx_a4500:1 \
     --mem=28G \
     -t 60 \
     --pty bash --login 