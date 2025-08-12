import argparse
import os, sys
import json
import math
import numpy as np
import copy

# Set CUDA environment for deterministic behavior
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from mogaze_config_gated import config
from gated_attention_model import SiMLPeWithGatedGaze, create_gated_model
from mogaze_dataset_gaze import MoGazeGazeDataset, MoGazeGazeEval

# Import the fixed aggressive trainer
from aggressive_gate_trainer import AggressiveGateTrainer

# Fallback imports without logging
def simple_print_and_log(logger, message):
    # Clean message without emojis for Windows compatibility
    clean_message = message.encode('ascii', 'ignore').decode('ascii')
    print(clean_message)

def ensure_dir(path): 
    os.makedirs(path, exist_ok=True)

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--exp-name', type=str, default='aggressive_gate_clean.txt', help='experiment name')
parser.add_argument('--seed', type=int, default=999, help='random seed')
parser.add_argument('--with-normalization', action='store_true', help='use layernorm')
parser.add_argument('--num', type=int, default=48, help='num of blocks')
parser.add_argument('--pretrained-baseline', type=str, 
                   default='log_gated/snapshot/model-iter-40000.pth', 
                   help='path to pretrained baseline model')
parser.add_argument('--gate-lr-multiplier', type=float, default=10.0, 
                   help='maximum gate learning rate multiplier')
parser.add_argument('--gate-reg-weight', type=float, default=0.1, 
                   help='gate regularization weight')

args = parser.parse_args()

# Set deterministic training
torch.use_deterministic_algorithms(True)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Create TensorBoard writer
writer = SummaryWriter(log_dir=f'runs/aggressive_gate_clean_{args.seed}')

# Open log file
try:
    acc_log = open(args.exp_name, 'a', encoding='utf-8')
except:
    acc_log = open('aggressive_gate_clean.txt', 'a', encoding='utf-8')

# Configure model parameters
config.motion_mlp.with_normalization = args.with_normalization
config.motion_mlp.num_layers = args.num
config.cos_lr_total_iters = 40000  # 5k + 20k + 15k

acc_log.write('AGGRESSIVE GATE LEARNING BREAKTHROUGH TRAINING - CLEAN VERSION\n')
acc_log.write('=' * 80 + '\n')
acc_log.write(f'Strategy: Force gate to utilize proven 23.6mm gaze improvement\n')
acc_log.write(f'Key Changes: Shorter baseline (5k), Higher gate LR (x{args.gate_lr_multiplier}), Gate regularization\n')
acc_log.write(f'Seed: {args.seed}\n')
acc_log.write(f'Previous Best: 73.2mm (conservative), Target: <70mm (aggressive)\n')
acc_log.write('=' * 80 + '\n')

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

def update_lr_multistep_aggressive(nb_iter, total_iter, max_lr, min_lr, optimizer, aggressive_trainer=None):
    """Aggressive learning rate schedule with high gate learning rates"""
    if nb_iter > 30000:
        base_lr = 1e-5
    else:
        base_lr = 3e-4
        
    # Apply aggressive gate learning rates
    if aggressive_trainer:
        gate_lr_multiplier = aggressive_trainer.get_gate_learning_rate_multiplier()
        
        for param_group in optimizer.param_groups:
            if 'gaze_gate' in param_group.get('name', ''):
                # AGGRESSIVE: Much higher learning rate for gate
                param_group["lr"] = base_lr * gate_lr_multiplier
                if gate_lr_multiplier > 0:
                    print(f"Gate LR: {param_group['lr']:.2e} (x{gate_lr_multiplier:.1f})")
            else:
                param_group["lr"] = base_lr
    else:
        for param_group in optimizer.param_groups:
            param_group["lr"] = base_lr
            
    return optimizer, base_lr

def train_step_aggressive(mogaze_motion_input, mogaze_motion_target, model, optimizer, nb_iter, 
                         total_iter, max_lr, min_lr, aggressive_trainer=None):
    """Aggressive training step with gate regularization and high learning rates"""
    
    if config.deriv_input:
        b, n, c = mogaze_motion_input.shape
        mogaze_motion_input_ = mogaze_motion_input.clone()
        mogaze_motion_input_ = torch.matmul(
            dct_m[:, :, :config.motion.h36m_input_length], 
            mogaze_motion_input_.cuda()
        )
    else:
        mogaze_motion_input_ = mogaze_motion_input.clone()

    # Forward pass with attention information for advanced stages
    if aggressive_trainer and aggressive_trainer.current_stage != 'baseline':
        motion_pred, attention_info = model(
            mogaze_motion_input_.cuda(), 
            return_attention=True
        )
    else:
        motion_pred = model(mogaze_motion_input_.cuda())
        attention_info = None
        
    motion_pred = torch.matmul(idct_m[:, :config.motion.h36m_input_length, :], motion_pred)

    if config.deriv_output:
        offset = mogaze_motion_input[:, -1:, :63].cuda()
        motion_pred = motion_pred[:, :config.motion.h36m_target_length_train] + offset
    else:
        motion_pred = motion_pred[:, :config.motion.h36m_target_length_train]

    # Compute AGGRESSIVE loss with regularization
    if aggressive_trainer:
        loss, loss_info = aggressive_trainer.compute_aggressive_loss(
            motion_pred, mogaze_motion_target, attention_info
        )
    else:
        # Fallback standard loss
        b, n, c_target = mogaze_motion_target.shape
        mogaze_motion_target_pose = mogaze_motion_target[:, :, :63].cuda()
        
        joints = 21
        motion_pred_joints = motion_pred.reshape(b, n, joints, 3).reshape(-1, 3)
        mogaze_motion_target_joints = mogaze_motion_target_pose.reshape(b, n, joints, 3).reshape(-1, 3)
        
        loss = torch.mean(torch.norm(motion_pred_joints - mogaze_motion_target_joints, 2, 1))
        loss_info = {'total_loss': loss.item(), 'gate_weight': model.get_gate_weight()}

    # Backward pass with AGGRESSIVE gradient handling
    optimizer.zero_grad()
    loss.backward()
    
    # AGGRESSIVE: Gradient clipping but allow higher gate gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)  # Increased from 1.0
    
    # Monitor gate gradients
    if hasattr(model, 'gaze_gate') and model.gaze_gate.grad is not None:
        gate_grad_norm = model.gaze_gate.grad.norm().item()
        if aggressive_trainer:
            aggressive_trainer.gate_gradients.append(gate_grad_norm)
        if gate_grad_norm > 0.000001:  # Only print significant gradients
            print(f"Gate gradient norm: {gate_grad_norm:.6f}")
    
    optimizer.step()
    
    # Update learning rate with AGGRESSIVE multipliers
    optimizer, current_lr = update_lr_multistep_aggressive(
        nb_iter, total_iter, max_lr, min_lr, optimizer, aggressive_trainer
    )
    
    # Update aggressive trainer
    if aggressive_trainer:
        aggressive_trainer.step()
        
    # TensorBoard logging
    writer.add_scalar('Loss/Total', loss_info['total_loss'], nb_iter)
    writer.add_scalar('Gate/Weight', loss_info['gate_weight'], nb_iter)
    writer.add_scalar('LR/Base', current_lr, nb_iter)
    
    if 'gate_regularization_loss' in loss_info:
        writer.add_scalar('Loss/Gate_Regularization', loss_info['gate_regularization_loss'], nb_iter)

    return loss_info, optimizer, current_lr

def main():
    print("AGGRESSIVE GATE LEARNING BREAKTHROUGH TRAINING - CLEAN VERSION")
    print("=" * 80)
    print("MISSION: Force gate to extract proven 23.6mm gaze improvement")
    print("STRATEGY: Shorter baseline + Higher gate LR + Regularization")
    print("=" * 80)
    
    # Create gated model
    print("Creating Gated Cross-Attention Model...")
    model = create_gated_model(config)
    
    # Load pretrained baseline if available
    if args.pretrained_baseline and os.path.exists(args.pretrained_baseline):
        print(f"Loading pretrained baseline from {args.pretrained_baseline}")
        try:
            checkpoint = torch.load(args.pretrained_baseline, map_location='cpu', weights_only=False)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=True)
                print("Loaded progressive training checkpoint")
            else:
                model.load_state_dict(checkpoint, strict=True)
                print("Loaded model state dict")
        except Exception as e:
            print(f"Could not load pretrained model: {e}")
            print("Starting from scratch...")
    
    model.train()
    model.cuda()
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters()) / 1000000.0
    print(f"Model parameters: {total_params:.2f}M")
    print(f"Initial gate weight: {model.get_gate_weight():.6f}")
    
    # Setup dataset
    config.motion.h36m_target_length = config.motion.h36m_target_length_train
    dataset = MoGazeGazeDataset(config, 'train', config.data_aug)
    
    dataloader = DataLoader(dataset, batch_size=config.batch_size,
                           num_workers=config.num_workers, drop_last=True,
                           shuffle=True, pin_memory=True)
    
    # Setup AGGRESSIVE trainer
    print("Initializing AGGRESSIVE Gate Learning Strategy...")
    aggressive_trainer = AggressiveGateTrainer(model, config, logger=None)
    
    # Update aggressive parameters from command line
    aggressive_trainer.max_gate_lr_multiplier = args.gate_lr_multiplier
    aggressive_trainer.gate_regularization_weight = args.gate_reg_weight
    
    aggressive_trainer.prepare_model_for_stage()
    
    print("AGGRESSIVE training configuration:")
    for stage_name, stage_info in aggressive_trainer.stages.items():
        print(f"   {stage_name.upper()}: {stage_info['description']}")
    print(f"   Max Gate LR Multiplier: x{args.gate_lr_multiplier}")
    print(f"   Gate Regularization: {args.gate_reg_weight}")
    
    # Setup optimizer with separate parameter groups
    param_groups = [
        {'params': [p for n, p in model.named_parameters() if 'gaze_gate' not in n], 'name': 'main'},
        {'params': [model.gaze_gate], 'name': 'gaze_gate'}
    ]
    
    optimizer = torch.optim.Adam(param_groups,
                                lr=config.cos_lr_max,
                                weight_decay=config.weight_decay)
    
    # Setup logging directories
    ensure_dir(config.snapshot_dir.replace('log_gated', 'log_aggressive'))
    aggressive_snapshot_dir = config.snapshot_dir.replace('log_gated', 'log_aggressive')

    print("AGGRESSIVE GATE LEARNING BREAKTHROUGH TRAINING")
    print("=" * 80)
    print(f"MISSION: Extract proven 23.6mm gaze improvement")
    print(f"Current baseline: 73.3mm, Conservative gated: 73.2mm")
    print(f"Target: <70mm with aggressive gate utilization")
    print(f"Architecture: {total_params:.2f}M parameters")
    print(f"Seed: {args.seed}")

    ##### ------ AGGRESSIVE BREAKTHROUGH TRAINING ------- #####
    nb_iter = 0
    avg_loss = 0.0
    avg_lr = 0.0
    best_gate_weight = 0.0
    best_performance = float('inf')

    print("Starting AGGRESSIVE Gate Learning...")
    print(f"Total planned iterations: {config.cos_lr_total_iters:,}")

    # Track aggressive learning metrics
    gate_opening_milestones = [0.1, 0.2, 0.3, 0.4, 0.5]  # Gate weight milestones to celebrate
    achieved_milestones = []

    while (nb_iter + 1) < config.cos_lr_total_iters:
        
        for (mogaze_motion_input, mogaze_motion_target) in dataloader:
            
            loss_info, optimizer, current_lr = train_step_aggressive(
                mogaze_motion_input, mogaze_motion_target, model, optimizer, 
                nb_iter, config.cos_lr_total_iters, config.cos_lr_max, config.cos_lr_min,
                aggressive_trainer
            )
            
            avg_loss += loss_info['total_loss']
            avg_lr += current_lr
            
            # Track gate opening milestones
            current_gate = loss_info['gate_weight']
            for milestone in gate_opening_milestones:
                if milestone not in achieved_milestones and current_gate >= milestone:
                    achieved_milestones.append(milestone)
                    print(f"MILESTONE: Gate opened to {milestone:.1f}! Current: {current_gate:.4f}")

            if (nb_iter + 1) % config.print_every == 0:
                avg_loss = avg_loss / config.print_every
                avg_lr = avg_lr / config.print_every
                
                # AGGRESSIVE progress logging
                aggressive_trainer.log_aggressive_progress(loss_info, writer, nb_iter + 1)
                
                summary = aggressive_trainer.get_aggressive_summary()
                
                print(f"Iter {nb_iter + 1} - {summary['strategy']}")
                print(f"   Stage: {summary['current_stage'].upper()} | Progress: {summary['total_progress']}")
                print(f"   Gate: {summary['current_gate_weight']:.4f} (Best: {summary['best_gate_weight']:.4f})")
                print(f"   Gate LR Multiplier: {summary['gate_learning_rate_multiplier']:.1f} | Temp: {summary['temperature']:.2f}")
                print(f"   Avg Loss: {avg_loss:.4f} | Base LR: {avg_lr:.2e}")
                
                if 'gate_regularization_loss' in loss_info and loss_info['gate_regularization_loss'] > 0:
                    print(f"   Gate Reg Loss: {loss_info['gate_regularization_loss']:.4f}")
                    
                if summary['gate_stuck_counter'] > 50:
                    print(f"   WARNING: Gate stuck for {summary['gate_stuck_counter']} iterations - may need intervention")
                    
                avg_loss = 0
                avg_lr = 0

            if (nb_iter + 1) % config.save_every == 0:
                # Save AGGRESSIVE checkpoint
                checkpoint_path = aggressive_snapshot_dir + '/model-iter-' + str(nb_iter + 1) + '.pth'
                
                # Save with aggressive training state
                aggressive_trainer.save_checkpoint(
                    checkpoint_path, 
                    optimizer,
                    {
                        'iteration': nb_iter + 1,
                        'gate_weight': model.get_gate_weight(),
                        'best_gate_weight': aggressive_trainer.best_gate_weight,
                        'gate_stuck_counter': aggressive_trainer.gate_stuck_counter,
                        'achieved_milestones': achieved_milestones,
                        'strategy': 'AGGRESSIVE_GATE_LEARNING',
                        'args': vars(args)
                    }
                )
                
                print(f"AGGRESSIVE checkpoint saved: iteration {nb_iter + 1}")
                
                # Track best gate weight
                current_gate = model.get_gate_weight()
                if current_gate > best_gate_weight:
                    best_gate_weight = current_gate
                    best_gate_path = aggressive_snapshot_dir + '/model-best-gate-aggressive.pth'
                    torch.save(model.state_dict(), best_gate_path)
                    print(f"NEW BEST GATE: {best_gate_weight:.4f} - saved to best-gate checkpoint")
                    
                # Log training summary
                summary = aggressive_trainer.get_aggressive_summary()
                print("AGGRESSIVE Training Summary:")
                for key, value in summary.items():
                    if key != 'strategy':
                        print(f"   {key}: {value}")

            # Check if AGGRESSIVE training is complete
            if aggressive_trainer.is_training_complete():
                print("AGGRESSIVE training completed!")
                break
                
            if (nb_iter + 1) == config.cos_lr_total_iters:
                break
                
            nb_iter += 1

    # Final AGGRESSIVE training summary
    writer.close()
    print("AGGRESSIVE GATE LEARNING COMPLETED!")
    print("=" * 80)
    
    final_summary = aggressive_trainer.get_aggressive_summary()
    print("FINAL AGGRESSIVE RESULTS:")
    print(f"   Final Gate Weight: {final_summary['current_gate_weight']:.4f}")
    print(f"   Best Gate Weight: {final_summary['best_gate_weight']:.4f}")
    print(f"   Achieved Milestones: {achieved_milestones}")
    print(f"   Gate Learning Success: {'YES' if final_summary['best_gate_weight'] > 0.1 else 'PARTIAL'}")
    
    # Calculate gate improvement
    gate_improvement = (final_summary['best_gate_weight'] - 0.0001) / 0.0001 * 100
    print(f"   Gate Opening: {gate_improvement:.1f}x improvement")
    
    # Save final model
    final_model_path = aggressive_snapshot_dir + f'/model-final-aggressive.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved: {final_model_path}")
    
    print("=" * 80)
    print("NEXT STEPS:")
    print("   1. Evaluate performance: python test_mogaze_gated.py --model-pth [best_checkpoint]")
    print("   2. Compare with baseline: Should see <73.2mm MPJPE if gate learned")
    print("   3. Analyze attention patterns with --analyze-attention flag")
    print(f"   4. Best checkpoint: {aggressive_snapshot_dir}/model-best-gate-aggressive.pth")
    
    if final_summary['best_gate_weight'] > 0.2:
        print("BREAKTHROUGH LIKELY ACHIEVED! Gate weight >0.2 suggests significant gaze utilization!")
    elif final_summary['best_gate_weight'] > 0.1:
        print("PARTIAL SUCCESS: Gate opened significantly, check evaluation results!")
    else:
        print("LIMITED OPENING: Consider even more aggressive strategies or different gaze features")
    
    acc_log.close()

if __name__ == "__main__":
    main()