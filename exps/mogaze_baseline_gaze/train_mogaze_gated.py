import argparse
import os, sys
import json
import math
import numpy as np
import copy

# FIXED: Set CUDA environment variable for deterministic behavior
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

try:
    from mogaze_config_gaze import config
    from gated_attention_model import SiMLPeWithGatedGaze, create_gated_model
    from progressive_trainer import ProgressiveGazeTrainer, GateMonitor
    from mogaze_dataset_gaze import MoGazeGazeDataset, MoGazeGazeEval
    print("All imports successful!")
except ImportError as e:
    print(f"Import error: {e}")
    print("Current directory:", os.getcwd())
    print("Files in directory:", [f for f in os.listdir('.') if f.endswith('.py')])
    sys.exit(1)

# Try to import utils - fallback if not available
try:
    from utils.logger import get_logger, print_and_log_info
    from utils.pyt_utils import link_file, ensure_dir
except ImportError:
    print("Utils not found - using fallback functions")
    
    def get_logger(path, name):
        import logging
        logger = logging.getLogger(name)
        handler = logging.FileHandler(path, encoding='utf-8')
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger
    
    def print_and_log_info(logger, message):
        print(message)
        if logger:
            logger.info(message)
    
    def link_file(src, dst):
        pass
    
    def ensure_dir(path):
        os.makedirs(path, exist_ok=True)

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--exp-name', type=str, default='mogaze_gated_attention.txt', help='=exp name')
parser.add_argument('--seed', type=int, default=888, help='=seed')
parser.add_argument('--temporal-only', action='store_true', help='=temporal only')
parser.add_argument('--layer-norm-axis', type=str, default='spatial', help='=layernorm axis')
parser.add_argument('--with-normalization', action='store_true', help='=use layernorm')
parser.add_argument('--spatial-fc', action='store_true', help='=use only spatial fc')
parser.add_argument('--num', type=int, default=48, help='=num of blocks')
parser.add_argument('--weight', type=float, default=1., help='=loss weight')
parser.add_argument('--progressive', action='store_true', help='=use progressive training')
parser.add_argument('--pretrained-pose', type=str, default=None, help='=path to pretrained pose-only model')

args = parser.parse_args()

torch.use_deterministic_algorithms(True)
torch.manual_seed(args.seed)
writer = SummaryWriter(log_dir=f'runs/gated_attention_{args.seed}')

# FIXED: Open log file with UTF-8 encoding
try:
    acc_log = open(args.exp_name, 'a', encoding='utf-8')
except:
    acc_log = open('gated_training_log.txt', 'a', encoding='utf-8')

# Configure model parameters
config.motion_fc_in.temporal_fc = args.temporal_only
config.motion_fc_out.temporal_fc = args.temporal_only
config.motion_mlp.norm_axis = args.layer_norm_axis
config.motion_mlp.spatial_fc_only = args.spatial_fc
config.motion_mlp.with_normalization = args.with_normalization
config.motion_mlp.num_layers = args.num

acc_log.write('GATED CROSS-ATTENTION BREAKTHROUGH TRAINING\n')
acc_log.write('=' * 60 + '\n')
acc_log.write(f'Seed: {args.seed}\n')
acc_log.write(f'Progressive Training: {args.progressive}\n')
acc_log.write(f'Target: Extract proven 23mm improvement from gaze\n')
acc_log.write('=' * 60 + '\n')

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

dct_m, idct_m = get_dct_matrix(config.motion.h36m_input_length_dct)
dct_m = torch.tensor(dct_m).float().cuda().unsqueeze(0)
idct_m = torch.tensor(idct_m).float().cuda().unsqueeze(0)

def update_lr_multistep(nb_iter, total_iter, max_lr, min_lr, optimizer, progressive_trainer=None):
    """Updated learning rate schedule with progressive training support"""
    if nb_iter > 30000:
        base_lr = 1e-5
    else:
        base_lr = 3e-4
        
    # Apply different learning rates based on training stage
    if progressive_trainer:
        gate_lr_multiplier = progressive_trainer.get_gate_learning_rate_multiplier()
        
        for param_group in optimizer.param_groups:
            if 'gaze_gate' in param_group.get('name', ''):
                # Special learning rate for gate parameter
                param_group["lr"] = base_lr * gate_lr_multiplier * 0.1  # Slower gate learning
            else:
                # Regular learning rate for other parameters
                param_group["lr"] = base_lr
    else:
        for param_group in optimizer.param_groups:
            param_group["lr"] = base_lr
            
    return optimizer, base_lr

def train_step_gated(mogaze_motion_input, mogaze_motion_target, model, optimizer, nb_iter, 
                    total_iter, max_lr, min_lr, progressive_trainer=None):
    """
    BREAKTHROUGH: Gated cross-attention training step
    """
    
    if config.deriv_input:
        b, n, c = mogaze_motion_input.shape  # c = 66 (pose+gaze)
        mogaze_motion_input_ = mogaze_motion_input.clone()
        mogaze_motion_input_ = torch.matmul(
            dct_m[:, :, :config.motion.h36m_input_length], 
            mogaze_motion_input_.cuda()
        )
    else:
        mogaze_motion_input_ = mogaze_motion_input.clone()

    # Forward pass with attention information
    if progressive_trainer and progressive_trainer.current_stage != 'baseline':
        # Get attention information for advanced stages
        motion_pred, attention_info = model(
            mogaze_motion_input_.cuda(), 
            return_attention=True
        )
    else:
        # Simple forward pass for baseline stage
        motion_pred = model(mogaze_motion_input_.cuda())
        attention_info = None
        
    motion_pred = torch.matmul(idct_m[:, :config.motion.h36m_input_length, :], motion_pred)

    if config.deriv_output:
        # Use only pose component of input for residual (first 63 dimensions)
        offset = mogaze_motion_input[:, -1:, :63].cuda()  # Extract pose only
        motion_pred = motion_pred[:, :config.motion.h36m_target_length_train] + offset
    else:
        motion_pred = motion_pred[:, :config.motion.h36m_target_length_train]

    # Compute progressive loss
    if progressive_trainer:
        loss, loss_info = progressive_trainer.compute_progressive_loss(
            motion_pred, mogaze_motion_target, attention_info
        )
    else:
        # Fallback to standard loss
        b, n, c_target = mogaze_motion_target.shape
        mogaze_motion_target_pose = mogaze_motion_target[:, :, :63].cuda()
        
        joints = 21
        motion_pred_joints = motion_pred.reshape(b, n, joints, 3).reshape(-1, 3)
        mogaze_motion_target_joints = mogaze_motion_target_pose.reshape(b, n, joints, 3).reshape(-1, 3)
        
        loss = torch.mean(torch.norm(motion_pred_joints - mogaze_motion_target_joints, 2, 1))
        loss_info = {'total_loss': loss.item(), 'gate_weight': model.get_gate_weight()}

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    # Update learning rate
    optimizer, current_lr = update_lr_multistep(
        nb_iter, total_iter, max_lr, min_lr, optimizer, progressive_trainer
    )
    
    # Update progressive trainer
    if progressive_trainer:
        progressive_trainer.step()
        
    # Log to tensorboard
    writer.add_scalar('Loss/Total', loss_info['total_loss'], nb_iter)
    writer.add_scalar('Gate/Weight', loss_info['gate_weight'], nb_iter)
    writer.add_scalar('LR/Base', current_lr, nb_iter)
    
    if 'alignment_loss' in loss_info:
        writer.add_scalar('Loss/Alignment', loss_info['alignment_loss'], nb_iter)

    return loss_info, optimizer, current_lr

def main():
    print("GATED CROSS-ATTENTION BREAKTHROUGH TRAINING")
    print("=" * 80)
    
    # Load pretrained pose model if specified
    pretrained_pose_model = None
    if args.pretrained_pose:
        if os.path.exists(args.pretrained_pose):
            print(f"Loading pretrained pose model from {args.pretrained_pose}")
            try:
                from model import siMLPe as BaseModel
                pretrained_pose_model = BaseModel(config)
                pretrained_state = torch.load(args.pretrained_pose, weights_only=True)
                pretrained_pose_model.load_state_dict(pretrained_state, strict=True)
                print("Pretrained pose model loaded successfully")
            except Exception as e:
                print(f"Failed to load pretrained model: {e}")
        else:
            print(f"Pretrained model not found: {args.pretrained_pose}")

    # Create gated cross-attention model
    print("Creating Gated Cross-Attention Model...")
    model = create_gated_model(config, pretrained_pose_model)
    model.train()
    model.cuda()

    # Model information
    total_params = sum(p.numel() for p in model.parameters()) / 1000000.0
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000.0
    print(f"Total parameters: {total_params:.2f}M")
    print(f"Trainable parameters: {trainable_params:.2f}M")
    print(f"Initial gate weight: {model.get_gate_weight():.6f}")

    # Setup dataset
    config.motion.h36m_target_length = config.motion.h36m_target_length_train
    dataset = MoGazeGazeDataset(config, 'train', config.data_aug)

    dataloader = DataLoader(dataset, batch_size=config.batch_size,
                            num_workers=config.num_workers, drop_last=True,
                            shuffle=True, pin_memory=True)

    # Setup progressive trainer
    progressive_trainer = None
    gate_monitor = GateMonitor()

    if args.progressive:
        print("Initializing Progressive Training Strategy...")
        progressive_trainer = ProgressiveGazeTrainer(model, config, logger=None)
        progressive_trainer.prepare_model_for_stage()
        
        print("Progressive training stages configured:")
        for stage_name, stage_info in progressive_trainer.stages.items():
            print(f"   {stage_name.upper()}: {stage_info['description']}")

    # Setup optimizer with parameter groups for different learning rates
    param_groups = [
        {'params': [p for n, p in model.named_parameters() if 'gaze_gate' not in n], 'name': 'main'},
        {'params': [model.gaze_gate], 'name': 'gaze_gate'}
    ]

    optimizer = torch.optim.Adam(param_groups,
                                 lr=config.cos_lr_max,
                                 weight_decay=config.weight_decay)

    # Setup logging
    ensure_dir(config.snapshot_dir)
    logger = get_logger(config.log_file.replace('log_gaze', 'log_gated'), 'train')

    print_and_log_info(logger, "GATED CROSS-ATTENTION BREAKTHROUGH TRAINING")
    print_and_log_info(logger, "=" * 80)
    print_and_log_info(logger, f"Target: Extract 23mm improvement from proven gaze information")
    print_and_log_info(logger, f"Baseline to beat: 73.3mm MPJPE")
    print_and_log_info(logger, f"Architecture: Gated cross-attention with {total_params:.2f}M parameters")
    print_and_log_info(logger, f"Progressive Training: {args.progressive}")

    ##### ------ BREAKTHROUGH TRAINING ------- #####
    nb_iter = 0
    avg_loss = 0.
    avg_lr = 0.
    best_gate_weight = 0.0

    print_and_log_info(logger, "Starting Gated Cross-Attention Training...")

    # Determine total iterations
    if progressive_trainer:
        total_iterations = sum(stage['iterations'] for stage in progressive_trainer.stages.values())
    else:
        total_iterations = config.cos_lr_total_iters

    print_and_log_info(logger, f"Total planned iterations: {total_iterations}")

    while (nb_iter + 1) < total_iterations:
        
        for (mogaze_motion_input, mogaze_motion_target) in dataloader:
            
            loss_info, optimizer, current_lr = train_step_gated(
                mogaze_motion_input, mogaze_motion_target, model, optimizer, 
                nb_iter, total_iterations, config.cos_lr_max, config.cos_lr_min,
                progressive_trainer
            )
            
            # Update monitoring
            gate_monitor.update(loss_info['gate_weight'], loss_info['total_loss'])
            
            avg_loss += loss_info['total_loss']
            avg_lr += current_lr

            if (nb_iter + 1) % config.print_every == 0:
                avg_loss = avg_loss / config.print_every
                avg_lr = avg_lr / config.print_every
                
                if progressive_trainer:
                    # Progressive training logging
                    progressive_trainer.log_progress(loss_info, writer, nb_iter + 1)
                    
                    stage_info = progressive_trainer.get_current_stage_info()
                    progress_pct = stage_info['progress'] * 100
                    
                    print_and_log_info(logger, f"Iter {nb_iter + 1} - Stage: {stage_info['current_stage'].upper()}")
                    print_and_log_info(logger, f"   Progress: {progress_pct:.1f}% | Gate: {loss_info['gate_weight']:.4f}")
                    print_and_log_info(logger, f"   Avg Loss: {avg_loss:.4f} | LR: {avg_lr:.2e}")
                    
                    if 'alignment_loss' in loss_info:
                        print_and_log_info(logger, f"   Alignment Loss: {loss_info['alignment_loss']:.4f}")
                        
                else:
                    print_and_log_info(logger, f"Iter {nb_iter + 1} Summary:")
                    print_and_log_info(logger, f"   Gate Weight: {loss_info['gate_weight']:.4f}")
                    print_and_log_info(logger, f"   Avg Loss: {avg_loss:.4f} | LR: {avg_lr:.2e}")
                    
                # Gate learning statistics
                gate_stats = gate_monitor.get_statistics()
                if gate_stats:
                    print_and_log_info(logger, f"   Gate Stats: mean={gate_stats['mean_gate']:.4f}, "
                                             f"std={gate_stats['gate_std']:.4f}, "
                                             f"velocity={gate_stats['gate_velocity']:.6f}")
                    
                avg_loss = 0
                avg_lr = 0

            if (nb_iter + 1) % config.save_every == 0:
                # Save model checkpoint
                checkpoint_path = config.snapshot_dir + '/model-iter-' + str(nb_iter + 1) + '.pth'
                
                if progressive_trainer:
                    # Save with progressive training state
                    progressive_trainer.save_checkpoint(
                        checkpoint_path, 
                        optimizer,
                        {
                            'iteration': nb_iter + 1,
                            'gate_weight': model.get_gate_weight(),
                            'gate_stats': gate_monitor.get_statistics()
                        }
                    )
                    
                    training_summary = progressive_trainer.get_training_summary()
                    print_and_log_info(logger, f"Training Summary:")
                    for key, value in training_summary.items():
                        print_and_log_info(logger, f"   {key}: {value}")
                        
                else:
                    torch.save(model.state_dict(), checkpoint_path)
                    
                print_and_log_info(logger, f"Model saved at iteration {nb_iter + 1}")
                
                # Log current gate weight as potential breakthrough indicator
                current_gate = model.get_gate_weight()
                if current_gate > best_gate_weight:
                    best_gate_weight = current_gate
                    print_and_log_info(logger, f"NEW BEST GATE WEIGHT: {best_gate_weight:.4f}")
                    
                    # Save best gate checkpoint
                    best_path = config.snapshot_dir + '/model-best-gate.pth'
                    torch.save(model.state_dict(), best_path)

            # Check if training is complete
            if progressive_trainer and progressive_trainer.is_training_complete():
                print_and_log_info(logger, "Progressive training completed!")
                break
                
            if (nb_iter + 1) == total_iterations:
                break
                
            nb_iter += 1

    # Final training summary
    writer.close()
    print_and_log_info(logger, "GATED CROSS-ATTENTION TRAINING COMPLETED!")
    print_and_log_info(logger, "=" * 80)
    print_and_log_info(logger, f"Final model: {config.snapshot_dir}/model-iter-{nb_iter + 1}.pth")
    print_and_log_info(logger, f"Final gate weight: {model.get_gate_weight():.4f}")
    print_and_log_info(logger, f"Best gate weight achieved: {best_gate_weight:.4f}")

    final_gate_stats = gate_monitor.get_statistics()
    if final_gate_stats:
        print_and_log_info(logger, f"Final gate statistics:")
        for key, value in final_gate_stats.items():
            print_and_log_info(logger, f"   {key}: {value}")

    if progressive_trainer:
        final_summary = progressive_trainer.get_training_summary()
        print_and_log_info(logger, f"Final training state:")
        for key, value in final_summary.items():
            print_and_log_info(logger, f"   {key}: {value}")

    print_and_log_info(logger, "Next: Run test_mogaze_gated.py to evaluate breakthrough performance!")
    acc_log.close()

if __name__ == "__main__":
    main()