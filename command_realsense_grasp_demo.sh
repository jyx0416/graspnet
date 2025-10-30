#!/usr/bin/env bash
# Use explicit python from the conda environment that has the required packages
# PYTHON=/home/jyx/miniconda3/envs/graspnet_baseline/bin/python
CUDA_VISIBLE_DEVICES=0 python grasp_process.py --checkpoint_path checkpoint/checkpoint-rs.tar
