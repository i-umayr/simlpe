import argparse
import os, sys
import json
import math
import numpy as np
import copy

# FIXED: Set CUDA environment variable for deterministic behavior
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

from mogaze_config_gaze import config
from model import siMLPe as Model
from mogaze_dataset_gaze import MoGazeGazeDataset
from utils.logger import get_logger, print_and_log_info
from utils.pyt_utils import link_file, ensure_dir
from mogaze_dataset_gaze import MoGazeGazeEval

# FIXED: Remove the test import to avoid the 5000 iteration crash
# from test_mogaze_gaze import test

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--exp-name', type=str, default='mogaze_gaze_experiment.txt', help='=exp name')
parser.add_argument('--seed', type=int, default=888, help='=seed')
parser.add_argument('--temporal-only', action='store_true', help='=temporal only')
parser.add_argument('--layer-norm-axis', type=str, default='spatial', help='=layernorm axis')
parser.add_argument('--with-normalization', action='store_true', help='=use layernorm')
parser.add_argument('--spatial-fc', action='store_true', help='=use only spatial fc')
parser.add_argument('--num', type=int, default=48, help='=num of blocks')
parser.add_argument('--weight', type=float, default=1., help='=loss weight')

args = parser.parse_args()

torch.use_deterministic_algorithms(True)
acc_log = open(args.exp_name, 'a')
torch.manual_seed(args.seed)
writer = SummaryWriter()

config.motion_fc_in.temporal_fc = args.temporal_only
config.motion_fc_out.temporal_fc = args.temporal_only
config.motion_mlp.norm_axis = args.layer_norm_axis
config.motion_mlp.spatial_fc_only = args.spatial_fc
config.motion_mlp.with_normalization = args.with_normalization
config.motion_mlp.num_layers = args.num

acc_log.write('Gaze Integration (Input Only) - Seed : ' + str(args.seed) + '\n')

def get_dct_matrix(N):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m

dct_m,idct_m = get_dct_matrix(config.motion.h36m_input_length_dct)
dct_m = torch.tensor(dct_m).float().cuda().unsqueeze(0)
idct_m = torch.tensor(idct_m).float().cuda().unsqueeze(0)

def update_lr_multistep(nb_iter, total_iter, max_lr, min_lr, optimizer) :
    if nb_iter > 30000:
        current_lr = 1e-5
    else:
        current_lr = 3e-4

    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr

def gen_velocity(m):
    dm = m[:, 1:] - m[:, :-1]
    return dm

def train_step(mogaze_motion_input, mogaze_motion_target, model, optimizer, nb_iter, total_iter, max_lr, min_lr):
    """
    CORRECTED: 
    - Input: 66D (63 pose + 3 gaze)
    - Target: Extract 63D pose only from 66D target
    - Model output: 63D (pose only)
    """
    
    if config.deriv_input:
        b,n,c = mogaze_motion_input.shape  # c = 66 (pose+gaze)
        mogaze_motion_input_ = mogaze_motion_input.clone()
        mogaze_motion_input_ = torch.matmul(dct_m[:, :, :config.motion.h36m_input_length], mogaze_motion_input_.cuda())
    else:
        mogaze_motion_input_ = mogaze_motion_input.clone()

    # Forward pass: 66D input → 63D output (pose only)
    motion_pred = model(mogaze_motion_input_.cuda())  # [batch, frames, 63]
    motion_pred = torch.matmul(idct_m[:, :config.motion.h36m_input_length, :], motion_pred)

    if config.deriv_output:
        # FIXED: Use only pose component of input for residual (first 63 dimensions)
        offset = mogaze_motion_input[:, -1:, :63].cuda()  # Extract pose only
        motion_pred = motion_pred[:, :config.motion.h36m_target_length_train] + offset
    else:
        motion_pred = motion_pred[:, :config.motion.h36m_target_length_train]

    b,n,c_target = mogaze_motion_target.shape  # c_target = 66 (pose+gaze from dataset)
    
    # CRITICAL FIX: Extract only pose component from target (first 63 dimensions)
    mogaze_motion_target_pose = mogaze_motion_target[:, :, :63].cuda()  # Extract pose only
    
    # Now both prediction and target are 63D (pose only)
    joints = 21  # MoGaze has 21 joints
    motion_pred = motion_pred.reshape(b, n, joints, 3).reshape(-1, 3)  # 63D → joints
    mogaze_motion_target_pose = mogaze_motion_target_pose.reshape(b, n, joints, 3).reshape(-1, 3)  # 63D target
    
    # Standard MPJPE loss on pose only
    loss = torch.mean(torch.norm(motion_pred - mogaze_motion_target_pose, 2, 1))

    if config.use_relative_loss:
        motion_pred_full = motion_pred.reshape(b, n, joints, 3)
        dmotion_pred = gen_velocity(motion_pred_full)
        motion_gt_full = mogaze_motion_target_pose.reshape(b, n, joints, 3)
        dmotion_gt = gen_velocity(motion_gt_full)
        
        # Velocity loss on pose only (63D)
        dloss = torch.mean(torch.norm((dmotion_pred - dmotion_gt).reshape(-1, 3), 2, 1))
        loss = loss + dloss
    else:
        loss = loss.mean()

    writer.add_scalar('Loss/total', loss.detach().cpu().numpy(), nb_iter)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    optimizer, current_lr = update_lr_multistep(nb_iter, total_iter, max_lr, min_lr, optimizer)
    writer.add_scalar('LR/train', current_lr, nb_iter)

    return loss.item(), optimizer, current_lr

model = Model(config)
model.train()
model.cuda()

config.motion.h36m_target_length = config.motion.h36m_target_length_train
dataset = MoGazeGazeDataset(config, 'train', config.data_aug)

shuffle = True
sampler = None
dataloader = DataLoader(dataset, batch_size=config.batch_size,
                        num_workers=config.num_workers, drop_last=True,
                        sampler=sampler, shuffle=shuffle, pin_memory=True)

# SKIP EVALUATION DURING TRAINING to avoid crashes
# eval_config = copy.deepcopy(config)
# eval_config.motion.h36m_target_length = eval_config.motion.h36m_target_length_eval
# eval_dataset = MoGazeGazeEval(eval_config, 'test')

# shuffle = False
# sampler = None
# eval_dataloader = DataLoader(eval_dataset, batch_size=128,
#                         num_workers=0, drop_last=False,
#                         sampler=sampler, shuffle=shuffle, pin_memory=True)

# initialize optimizer
optimizer = torch.optim.Adam(model.parameters(),
                             lr=config.cos_lr_max,
                             weight_decay=config.weight_decay)

ensure_dir(config.snapshot_dir)
logger = get_logger(config.log_file, 'train')
link_file(config.log_file, config.link_log_file)

print_and_log_info(logger, "MoGaze Gaze Integration Training (CORRECTED)")
print_and_log_info(logger, f"Input dimensions: 66D (63 pose + 3 gaze)")
print_and_log_info(logger, f"Output dimensions: 63D (pose only)")
print_and_log_info(logger, f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1000000.0:.2f}M")
print_and_log_info(logger, json.dumps(config, indent=4, sort_keys=True))

if config.model_pth is not None :
    state_dict = torch.load(config.model_pth)
    model.load_state_dict(state_dict, strict=True)
    print_and_log_info(logger, "Loading model path from {} ".format(config.model_pth))

##### ------ training ------- #####
nb_iter = 0
avg_loss = 0.
avg_lr = 0.

print_and_log_info(logger, "Starting gaze integration training...")
print_and_log_info(logger, f"Baseline to beat: 73.3 mm MPJPE (3D position only)")
print_and_log_info(logger, f"Approach: Use gaze as input to predict better poses")

while (nb_iter + 1) < config.cos_lr_total_iters:

    for (mogaze_motion_input, mogaze_motion_target) in dataloader:

        loss, optimizer, current_lr = train_step(mogaze_motion_input, mogaze_motion_target, model, optimizer, nb_iter, config.cos_lr_total_iters, config.cos_lr_max, config.cos_lr_min)
        avg_loss += loss
        avg_lr += current_lr

        if (nb_iter + 1) % config.print_every ==  0 :
            avg_loss = avg_loss / config.print_every
            avg_lr = avg_lr / config.print_every

            print_and_log_info(logger, "Iter {} Summary: ".format(nb_iter + 1))
            print_and_log_info(logger, f"\t lr: {avg_lr} \t Training loss: {avg_loss}")
            avg_loss = 0
            avg_lr = 0

        if (nb_iter + 1) % config.save_every ==  0 :
            torch.save(model.state_dict(), config.snapshot_dir + '/model-iter-' + str(nb_iter + 1) + '.pth')
            print_and_log_info(logger, f"Model saved at iteration {nb_iter + 1}")
            # REMOVED: evaluation call that was causing crashes

        if (nb_iter + 1) == config.cos_lr_total_iters :
            break
        nb_iter += 1

writer.close()
print_and_log_info(logger, "Gaze integration training completed!")
print_and_log_info(logger, f"Final model saved at: {config.snapshot_dir}/model-iter-{config.cos_lr_total_iters}.pth")
print_and_log_info(logger, "Next: Run test_mogaze_gaze.py to evaluate against 73.3mm baseline")