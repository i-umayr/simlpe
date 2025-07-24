# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict
import argparse

C = edict()
config = C
cfg = C

C.seed = 304

"""please config ROOT_dir and user when u first using"""
C.abs_dir = osp.dirname(osp.realpath(__file__))
C.this_dir = C.abs_dir.split(osp.sep)[-1]
C.repo_name = 'siMLPe'
C.root_dir = C.abs_dir[:C.abs_dir.index(C.repo_name) + len(C.repo_name)]

C.log_dir = osp.abspath(osp.join(C.abs_dir, 'log_3d'))
C.snapshot_dir = osp.abspath(osp.join(C.log_dir, "snapshot"))

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.log_file = C.log_dir + '/log_' + exp_time + '.log'
C.link_log_file = C.log_dir + '/log_last.log'
C.val_log_file = C.log_dir + '/val_' + exp_time + '.log'
C.link_val_log_file = C.log_dir + '/val_last.log'

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(osp.join(C.root_dir, 'lib'))

"""Data Dir and Weight Dir"""
# MoGaze 3D position dataset configuration
C.mogaze_anno_dir = osp.join(C.root_dir, 'data/mogaze_3d/')

"""Dataset Config"""
C.motion = edict()

# MoGaze motion parameters (matching siMLPe H36M approach)
C.motion.mogaze_input_length = 50      # 50 input frames (same as H36M)
C.motion.mogaze_input_length_dct = 50  # DCT input length

# FIXED: These need to match the H36M naming convention for the model to work
C.motion.h36m_input_length = 50        # Model expects this name
C.motion.h36m_input_length_dct = 50    # Model expects this name
C.motion.h36m_target_length_train = 10 # Model expects this name
C.motion.h36m_target_length_eval = 30  # Model expects this name

# Keep MoGaze-specific names for dataset
C.motion.mogaze_target_length_train = 10  # 10 target frames for training (same as H36M)
C.motion.mogaze_target_length_eval = 30   # 30 target frames for evaluation (auto-regressive)

# UPDATED: MoGaze 3D positions = 21 joints × 3 coordinates = 63 dimensions (was 66 for Euler)
C.motion.dim = 63

# Data augmentation and processing flags
C.data_aug = True
C.deriv_input = True
C.deriv_output = True
C.use_relative_loss = True

""" Model Config"""
## Network
C.pre_dct = False
C.post_dct = False

## Motion Network mlp
dim_ = 63  # UPDATED: 21 joints * 3 positions (was 66 for Euler angles)
C.motion_mlp = edict()
C.motion_mlp.hidden_dim = dim_
C.motion_mlp.seq_len = 50  # Input sequence length for MLP blocks
C.motion_mlp.num_layers = 48
C.motion_mlp.with_normalization = True
C.motion_mlp.spatial_fc_only = False
C.motion_mlp.norm_axis = 'spatial'

## Motion Network FC In
C.motion_fc_in = edict()
C.motion_fc_in.in_features = C.motion.dim
C.motion_fc_in.out_features = dim_
C.motion_fc_in.with_norm = False
C.motion_fc_in.activation = 'relu'
C.motion_fc_in.init_w_trunc_normal = False
C.motion_fc_in.temporal_fc = False

## Motion Network FC Out
C.motion_fc_out = edict()
C.motion_fc_out.in_features = dim_
C.motion_fc_out.out_features = C.motion.dim
C.motion_fc_out.with_norm = False
C.motion_fc_out.activation = 'relu'
C.motion_fc_out.init_w_trunc_normal = True
C.motion_fc_out.temporal_fc = False

"""Train Config"""
C.batch_size = 32  # Reduced from 256 to debug
C.num_workers = 0  # Set to 0 for Windows compatibility

# Learning rate configuration
C.cos_lr_max = 1e-5
C.cos_lr_min = 5e-8
C.cos_lr_total_iters = 40000

C.weight_decay = 1e-4
C.model_pth = None

"""Eval Config"""
C.shift_step = 1

"""Display Config"""
C.print_every = 100
C.save_every = 5000

if __name__ == '__main__':
    print("🎯 MoGaze 3D Position Configuration")
    print(f"Data dimensions: {C.motion.dim} (21 joints × 3 positions)")
    print(f"Hidden dimensions: {dim_}")
    print(f"Dataset directory: {C.mogaze_anno_dir}")
    print(f"Log directory: {C.log_dir}")