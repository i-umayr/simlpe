import argparse
import os, sys
import json
import math
import numpy as np
import copy

# Add multiprocessing fix for Windows
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

from mogaze_config_3d import config
from model import siMLPe as Model
from mogaze_dataset_3d import MoGaze3DDataset, MoGaze3DEval
from utils.logger import get_logger, print_and_log_info
from utils.pyt_utils import link_file, ensure_dir

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--exp-name', type=str, default='mogaze_3d_baseline.txt', help='experiment name')
parser.add_argument('--seed', type=int, default=888, help='random seed')
parser.add_argument('--temporal-only', action='store_true', help='temporal only')
parser.add_argument('--layer-norm-axis', type=str, default='spatial', help='layernorm axis')
parser.add_argument('--with-normalization', action='store_true', help='use layernorm')
parser.add_argument('--spatial-fc', action='store_true', help='use only spatial fc')
parser.add_argument('--num', type=int, default=48, help='number of blocks')
parser.add_argument('--weight', type=float, default=1., help='loss weight')

args = parser.parse_args()

torch.use_deterministic_algorithms(True)
acc_log = open(args.exp_name, 'a')
torch.manual_seed(args.seed)
writer = SummaryWriter()

# Update config based on arguments
config.motion_fc_in.temporal_fc = args.temporal_only
config.motion_fc_out.temporal_fc = args.temporal_only
config.motion_mlp.norm_axis = args.layer_norm_axis
config.motion_mlp.spatial_fc_only = args.spatial_fc
config.motion_mlp.with_normalization = args.with_normalization
config.motion_mlp.num_layers = args.num

acc_log.write('Seed : ' + str(args.seed) + '\n')

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

# Use the correct config attribute names that match H36M
dct_m, idct_m = get_dct_matrix(config.motion.h36m_input_length_dct)
dct_m = torch.tensor(dct_m).float().cuda().unsqueeze(0)
idct_m = torch.tensor(idct_m).float().cuda().unsqueeze(0)

def update_lr_multistep(nb_iter, total_iter, max_lr, min_lr, optimizer):
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
    
    if config.deriv_input:
        b, n, c = mogaze_motion_input.shape
        mogaze_motion_input_ = mogaze_motion_input.clone()
        # Use the correct config attribute name
        mogaze_motion_input_ = torch.matmul(dct_m[:, :, :config.motion.h36m_input_length], mogaze_motion_input_.cuda())
    else:
        mogaze_motion_input_ = mogaze_motion_input.clone()

    motion_pred = model(mogaze_motion_input_.cuda())
    # Use the correct config attribute name  
    motion_pred = torch.matmul(idct_m[:, :config.motion.h36m_input_length, :], motion_pred)

    if config.deriv_output:
        offset = mogaze_motion_input[:, -1:].cuda()
        # Use the correct config attribute name
        motion_pred = motion_pred[:, :config.motion.h36m_target_length_train] + offset
    else:
        # Use the correct config attribute name
        motion_pred = motion_pred[:, :config.motion.h36m_target_length_train]

    b, n, c = mogaze_motion_target.shape
    # For MoGaze 3D positions: 21 joints * 3 coordinates = 63 dimensions
    joints = 21  # MoGaze has 21 joints (same as GazeMotion)
    motion_pred = motion_pred.reshape(b, n, joints, 3).reshape(-1, 3)
    mogaze_motion_target = mogaze_motion_target.cuda().reshape(b, n, joints, 3).reshape(-1, 3)
    loss = torch.mean(torch.norm(motion_pred - mogaze_motion_target, 2, 1))

    if config.use_relative_loss:
        motion_pred = motion_pred.reshape(b, n, joints, 3)
        dmotion_pred = gen_velocity(motion_pred)
        motion_gt = mogaze_motion_target.reshape(b, n, joints, 3)
        dmotion_gt = gen_velocity(motion_gt)
        dloss = torch.mean(torch.norm((dmotion_pred - dmotion_gt).reshape(-1, 3), 2, 1))
        loss = loss + dloss
    else:
        loss = loss.mean()

    writer.add_scalar('Loss/mogaze_3d', loss.detach().cpu().numpy(), nb_iter)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    optimizer, current_lr = update_lr_multistep(nb_iter, total_iter, max_lr, min_lr, optimizer)
    writer.add_scalar('LR/train', current_lr, nb_iter)

    return loss.item(), optimizer, current_lr

# Initialize model
model = Model(config)
model.train()
model.cuda()

print("🚀 Training siMLPe on MoGaze 3D Positions")
print("=" * 50)
print(f"📊 Model configuration:")
print(f"   Input dimensions: {config.motion.dim} (21 joints × 3 positions)")
print(f"   Hidden dimensions: {config.motion_mlp.hidden_dim}")
print(f"   MLP layers: {config.motion_mlp.num_layers}")
print(f"   Dataset directory: {config.mogaze_anno_dir}")

# Training dataset
config.motion.h36m_target_length = config.motion.h36m_target_length_train
dataset = MoGaze3DDataset(config, 'train', config.data_aug)

shuffle = True
sampler = None
dataloader = DataLoader(dataset, batch_size=config.batch_size,
                        num_workers=0, drop_last=True,
                        sampler=sampler, shuffle=shuffle, pin_memory=True)

# Evaluation dataset
eval_config = copy.deepcopy(config)
eval_config.motion.h36m_target_length = eval_config.motion.h36m_target_length_eval
eval_dataset = MoGaze3DEval(eval_config, 'test')

shuffle = False
sampler = None
eval_dataloader = DataLoader(eval_dataset, batch_size=128,
                        num_workers=0, drop_last=False,
                        sampler=sampler, shuffle=shuffle, pin_memory=True)

# Initialize optimizer
optimizer = torch.optim.Adam(model.parameters(),
                             lr=config.cos_lr_max,
                             weight_decay=config.weight_decay)

ensure_dir(config.snapshot_dir)
logger = get_logger(config.log_file, 'train')
link_file(config.log_file, config.link_log_file)

print_and_log_info(logger, "Training siMLPe on MoGaze 3D Positions")
print_and_log_info(logger, json.dumps(config, indent=4, sort_keys=True))

if config.model_pth is not None:
    state_dict = torch.load(config.model_pth)
    model.load_state_dict(state_dict, strict=True)
    print_and_log_info(logger, "Loading model path from {} ".format(config.model_pth))

##### ------ training ------- #####
nb_iter = 0
avg_loss = 0.
avg_lr = 0.

print_and_log_info(logger, f"Starting MoGaze 3D training...")
print_and_log_info(logger, f"Dataset size: {len(dataset)} samples")
print_and_log_info(logger, f"Eval dataset size: {len(eval_dataset)} samples")

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

        if (nb_iter + 1) == config.cos_lr_total_iters :
            break
        nb_iter += 1

writer.close()
print("🎉 Training completed!")
print(f"📁 Model saved in: {config.snapshot_dir}")
print("\n🔄 Next steps:")
print("1. Run test_mogaze_3d.py to evaluate the model")
print("2. Compare MPJPE results with GazeMotion benchmarks")
print("3. Results should be directly comparable (both use 3D positions)")