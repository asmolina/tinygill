#!/usr/bin/env bash

#SBATCH --job-name=llm2clip
#SBATCH --partition=ais-gpu 
#SBATCH --gpus=1 
#SBATCH --nodes=1             
#SBATCH --mem=32GB 
#SBATCH --time=08:00:00 
#SBATCH --ntasks-per-node=1  
#SBATCH --output=/trinity/home/alina.smolina/sbatch_logs/%x@%A_%a.out 
#SBATCH --error=/trinity/home/alina.smolina/sbatch_logs/%x@%A_%a.err


source activate torch

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

cd airi/

srun python3 -u ./sbatch_run.py \
    --batch-size=48 \
    --projection-method='CLIP-like' \
    --lr-base=1e-3 \
    --lr-min=1e-4 \
    --lr-period=4200 \
    --loss-variant='both' \
    --accum-steps=5 \
    --num-img-tokens=4