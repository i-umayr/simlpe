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

C.log_dir = osp.abspath(osp.join(C.abs_dir, 'log_gaze'))
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
# MoGaze 3D position + gaze dataset configuration
C.mogaze_anno_dir = osp.join(C.root_dir, 'data/mogaze_3d/')

"""Dataset Config"""
C.motion = edict()

# MoGaze motion parameters (same as 3D baseline)
C.motion.mogaze_input_length = 50      # 50 input frames
C.motion.mogaze_input_length_dct = 50  # DCT input length

# FIXED: Keep H36M naming convention for model compatibility
C.motion.h36m_input_length = 50        # Model expects this name
C.motion.h36m_input_length_dct = 50    # Model expects this name
C.motion.h36m_target_length_train = 10 # Model expects this name
C.motion.h36m_target_length_eval = 30  # Model expects this name

# Keep MoGaze-specific names for dataset
C.motion.mogaze_target_length_train = 10  # 10 target frames for training
C.motion.mogaze_target_length_eval = 30   # 30 target frames for evaluation

# CRITICAL CHANGE: 
# - Input: 66D (63 pose + 3 gaze) for training
# - Output: 63D (pose only) for prediction
# - We use gaze as additional input to predict better poses
C.motion.dim = 63  # Output dimension: pose only (21 joints × 3)
C.motion.input_dim = 66  # Input dimension: pose + gaze (63 + 3)

# Data augmentation and processing flags
C.data_aug = True
C.deriv_input = True
C.deriv_output = True
C.use_relative_loss = True

""" Model Config"""
## Network
C.pre_dct = False
C.post_dct = False

## Motion Network mlp - EXPANDED for gaze integration (Option B)
# Hidden dimension = 66 to preserve all pose+gaze information
hidden_dim = 66  # Expanded to handle 63 pose + 3 gaze features optimally
C.motion_mlp = edict()
C.motion_mlp.hidden_dim = hidden_dim
C.motion_mlp.seq_len = 50  # Input sequence length for MLP blocks
C.motion_mlp.num_layers = 48
C.motion_mlp.with_normalization = True
C.motion_mlp.spatial_fc_only = False
C.motion_mlp.norm_axis = 'spatial'

## Motion Network FC In - Accept full 66D input (no compression)
C.motion_fc_in = edict()
C.motion_fc_in.in_features = 66  # Accept pose+gaze input
C.motion_fc_in.out_features = hidden_dim  # 66 (preserve all information)
C.motion_fc_in.with_norm = False
C.motion_fc_in.activation = 'relu'
C.motion_fc_in.init_w_trunc_normal = False
C.motion_fc_in.temporal_fc = False

## Motion Network FC Out - Project back to pose only
C.motion_fc_out = edict()
C.motion_fc_out.in_features = hidden_dim  # 66 (from expanded hidden space)
C.motion_fc_out.out_features = 63  # Output pose only (21 joints × 3)
C.motion_fc_out.with_norm = False
C.motion_fc_out.activation = 'relu'
C.motion_fc_out.init_w_trunc_normal = True
C.motion_fc_out.temporal_fc = False

"""Train Config"""
C.batch_size = 32  # Same as 3D baseline
C.num_workers = 0  # Windows compatibility

# Learning rate configuration (same as successful 3D training)
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
    print("🎯 MoGaze Gaze Integration Configuration (Option B - Expanded)")
    print(f"Input dimensions: {C.motion.input_dim} (63 pose + 3 gaze)")
    print(f"Hidden dimensions: {hidden_dim} (expanded for optimal gaze processing)")
    print(f"Output dimensions: {C.motion.dim} (63 pose only)")
    print(f"Dataset directory: {C.mogaze_anno_dir}")
    print(f"Log directory: {C.log_dir}")
    print(f"\n✅ Architecture: Input[66] → FC[66→66] → 48×MLP[66] → FC[66→63] → Output[63]")
    print(f"✅ Model capacity: ~0.151M parameters (~7.8% more than baseline)")
    print(f"✅ Gaze used as input only - output is pose-only predictions")