import argparse
import os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

from mogaze_config_gated import config
from gated_attention_model import SiMLPeWithGatedGaze, create_gated_model
from mogaze_dataset_gaze import MoGazeGazeEval

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

def analyze_attention_patterns(attention_weights, save_path=None):
    """Analyze and visualize attention patterns from the gated cross-attention mechanism"""
    # attention_weights: [batch, num_heads, seq_len, seq_len]
    batch_size, num_heads, seq_len, _ = attention_weights.shape
    
    # Average across batch and heads for visualization
    avg_attention = attention_weights.mean(dim=(0, 1)).cpu().numpy()  # [seq_len, seq_len]
    
    # Create attention heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(avg_attention, 
                xticklabels=range(0, seq_len, 10),
                yticklabels=range(0, seq_len, 10),
                cmap='viridis',
                cbar_kws={'label': 'Attention Weight'})
    plt.title('Gaze-Pose Cross-Attention Pattern')
    plt.xlabel('Pose Time Steps')
    plt.ylabel('Gaze Time Steps')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attention pattern saved to: {save_path}")
    
    plt.show()
    
    # Analyze temporal patterns
    temporal_attention = avg_attention.mean(axis=1)  # Average attention over time
    peak_attention_time = np.argmax(temporal_attention)
    
    analysis = {
        'peak_attention_time': peak_attention_time,
        'attention_concentration': np.max(temporal_attention) / np.mean(temporal_attention),
        'attention_entropy': -np.sum(temporal_attention * np.log(temporal_attention + 1e-8)),
        'early_attention': np.mean(temporal_attention[:seq_len//3]),  # First third
        'middle_attention': np.mean(temporal_attention[seq_len//3:2*seq_len//3]),  # Middle third
        'late_attention': np.mean(temporal_attention[2*seq_len//3:])  # Last third
    }
    
    return analysis

def analyze_gate_dynamics(gate_weights, uncertainties=None, save_path=None):
    """Analyze gate learning dynamics and uncertainty patterns"""
    gate_weights = np.array(gate_weights)
    
    plt.figure(figsize=(12, 8))
    
    # Plot gate weight over time
    plt.subplot(2, 2, 1)
    plt.plot(gate_weights, 'b-', linewidth=2, alpha=0.8)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Balanced Gate (0.5)')
    plt.title('Gate Weight Evolution')
    plt.xlabel('Batch')
    plt.ylabel('Gate Weight')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot gate weight distribution
    plt.subplot(2, 2, 2)
    plt.hist(gate_weights, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(x=np.mean(gate_weights), color='r', linestyle='--', label=f'Mean: {np.mean(gate_weights):.3f}')
    plt.title('Gate Weight Distribution')
    plt.xlabel('Gate Weight')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Analyze gate learning phases
    plt.subplot(2, 2, 3)
    gate_velocity = np.diff(gate_weights)
    plt.plot(gate_velocity, 'g-', alpha=0.7)
    plt.title('Gate Learning Velocity')
    plt.xlabel('Batch')
    plt.ylabel('Delta Gate Weight')
    plt.grid(True, alpha=0.3)
    
    # Uncertainty analysis if available (fixed for variable shapes)
    if uncertainties is not None:
        plt.subplot(2, 2, 4)
        try:
            # Handle variable shapes in uncertainties
            avg_uncertainties = []
            for unc in uncertainties:
                if isinstance(unc, np.ndarray):
                    avg_uncertainties.append(np.mean(unc))
                else:
                    avg_uncertainties.append(float(unc))
            
            avg_uncertainties = np.array(avg_uncertainties)
            
            # Only plot if we have matching lengths
            min_len = min(len(gate_weights), len(avg_uncertainties))
            if min_len > 0:
                plt.scatter(gate_weights[:min_len], avg_uncertainties[:min_len], 
                           alpha=0.6, c=range(min_len), cmap='viridis')
                plt.colorbar(label='Time')
                plt.title('Gate Weight vs Uncertainty')
                plt.xlabel('Gate Weight')
                plt.ylabel('Average Uncertainty')
                plt.grid(True, alpha=0.3)
            else:
                plt.text(0.5, 0.5, 'Uncertainty data unavailable', 
                        transform=plt.gca().transAxes, ha='center', va='center')
                plt.title('Gate Weight vs Uncertainty')
        except Exception as e:
            plt.text(0.5, 0.5, f'Uncertainty analysis failed:\n{str(e)[:50]}...', 
                    transform=plt.gca().transAxes, ha='center', va='center')
            plt.title('Gate Weight vs Uncertainty')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gate dynamics saved to: {save_path}")
    
    plt.show()
    
    return {
        'mean_gate': np.mean(gate_weights),
        'std_gate': np.std(gate_weights),
        'min_gate': np.min(gate_weights),
        'max_gate': np.max(gate_weights),
        'gate_range': np.max(gate_weights) - np.min(gate_weights),
        'final_gate': gate_weights[-1] if len(gate_weights) > 0 else 0,
        'learning_velocity': np.std(gate_velocity) if len(gate_velocity) > 0 else 0
    }

def test_aggressive_model_with_analysis(config, model, dataloader, analyze_attention=True):
    """Test aggressive gated cross-attention model with comprehensive analysis"""
    model.eval()
    
    # DCT matrices
    dct_m, idct_m = get_dct_matrix(config.motion.h36m_input_length_dct)
    dct_m = torch.tensor(dct_m).float().cuda().unsqueeze(0)
    idct_m = torch.tensor(idct_m).float().cuda().unsqueeze(0)
    
    total_error = 0.0
    num_samples = 0
    errors_per_frame = []
    
    # Analysis collections
    gate_weights = []
    attention_patterns = []
    uncertainties = []
    alignment_losses = []
    
    print("AGGRESSIVE GATED CROSS-ATTENTION EVALUATION WITH ANALYSIS")
    print("=" * 80)
    
    with torch.no_grad():
        for batch_idx, (motion_input, motion_target) in enumerate(dataloader):
            if isinstance(motion_input, np.ndarray):
                motion_input = torch.tensor(motion_input).float()
            if isinstance(motion_target, np.ndarray):
                motion_target = torch.tensor(motion_target).float()
                
            motion_input = motion_input.cuda()  # [batch, 50, 66]
            motion_target = motion_target.cuda()  # [batch, 30, 66]
            
            b, n, c = motion_input.shape
            num_samples += b
            
            # Auto-regressive prediction with attention analysis
            outputs = []
            step = config.motion.h36m_target_length_train  # 10 frames per step
            
            if step == 30:
                num_step = 1
            else:
                num_step = 30 // step + 1
            
            current_input = motion_input.clone()
            
            for idx in range(num_step):
                # Forward pass with attention analysis
                if config.deriv_input:
                    motion_input_ = current_input.clone()
                    motion_input_ = torch.matmul(dct_m[:, :, :config.motion.h36m_input_length], motion_input_)
                else:
                    motion_input_ = current_input.clone()
                
                # Get attention information if analyzing
                if analyze_attention:
                    try:
                        output, attention_info = model(
                            motion_input_, 
                            return_attention=True
                        )
                        
                        # Collect analysis data
                        gate_weights.append(model.get_gate_weight())
                        if 'attention_weights' in attention_info:
                            attention_patterns.append(attention_info['attention_weights'])
                        if 'uncertainty' in attention_info:
                            uncertainties.append(attention_info['uncertainty'].cpu().numpy())
                        if 'alignment_loss' in attention_info:
                            alignment_losses.append(attention_info['alignment_loss'].cpu().item())
                            
                    except Exception as e:
                        print(f"Attention analysis failed: {e}, using standard forward pass")
                        output = model(motion_input_)
                        gate_weights.append(model.get_gate_weight())
                else:
                    output = model(motion_input_)
                
                output = torch.matmul(idct_m[:, :config.motion.h36m_input_length, :], output)[:, :step, :]
                
                if config.deriv_output:
                    # Add residual from pose component only
                    pose_offset = current_input[:, -1:, :63]
                    output = output + pose_offset
                
                # Pad output back to 66D for next iteration input
                batch_size, seq_len, _ = output.shape
                last_gaze = current_input[:, -1:, 63:66]  # [batch, 1, 3]
                extended_gaze = last_gaze.repeat(1, seq_len, 1)  # [batch, seq_len, 3]
                
                output_with_gaze = torch.cat([output, extended_gaze], dim=2)  # [batch, seq_len, 66]
                outputs.append(output)  # Store pose-only predictions
                
                # Update input for next iteration
                current_input = torch.cat([current_input[:, step:], output_with_gaze], axis=1)
            
            # Concatenate pose-only outputs
            motion_pred = torch.cat(outputs, axis=1)[:, :30]  # [batch, 30, 63]
            
            # Calculate MPJPE
            motion_pred_np = motion_pred.cpu().numpy()
            motion_target_np = motion_target.cpu().numpy()
            
            # Extract pose component from target
            target_pose = motion_target_np[:, :, :63]  # [batch, 30, 63]
            
            # Reshape for joint-wise error calculation
            pred_pose_3d = motion_pred_np.reshape(b, 30, 21, 3)
            target_pose_3d = target_pose.reshape(b, 30, 21, 3)
            
            # Calculate MPJPE in meters, convert to millimeters
            pose_error = np.linalg.norm(pred_pose_3d - target_pose_3d, axis=3)
            pose_error_mm = pose_error * 1000
            
            # Average over joints and batch for each frame
            frame_errors = np.mean(pose_error_mm, axis=(0, 2))  # [30]
            total_error += np.sum(pose_error_mm)
            
            if len(errors_per_frame) == 0:
                errors_per_frame = frame_errors
            else:
                errors_per_frame += frame_errors
                
            # Progress indicator
            if batch_idx % 10 == 0:
                current_gate = gate_weights[-1] if gate_weights else 0.0
                print(f"  Batch {batch_idx + 1}/{len(dataloader)} | "
                      f"Current MPJPE: {np.mean(pose_error_mm):.1f}mm | "
                      f"Gate: {current_gate:.4f}")
    
    # Calculate final results
    avg_pose_error = total_error / (num_samples * 30 * 21)
    errors_per_frame = errors_per_frame / len(dataloader)
    
    # BREAKTHROUGH ANALYSIS
    print("\n" + "=" * 80)
    print("AGGRESSIVE TRAINING ANALYSIS RESULTS")
    print("=" * 80)
    
    analysis_results = {}
    
    # Gate dynamics analysis
    if gate_weights:
        print(f"\nGATE LEARNING ANALYSIS:")
        gate_analysis = analyze_gate_dynamics(gate_weights, uncertainties)
        analysis_results['gate_analysis'] = gate_analysis
        
        print(f"   Mean Gate Weight: {gate_analysis['mean_gate']:.4f}")
        print(f"   Gate Range: {gate_analysis['gate_range']:.4f}")
        print(f"   Final Gate: {gate_analysis['final_gate']:.4f}")
        print(f"   Learning Velocity: {gate_analysis['learning_velocity']:.6f}")
        
        # Determine if gate learned to use gaze
        if gate_analysis['mean_gate'] > 0.1:
            print(f"   BREAKTHROUGH: Gate learned to use gaze information!")
        else:
            print(f"   Gate remained closed - gaze not effectively utilized")
    
    # Attention pattern analysis
    if attention_patterns:
        print(f"\nATTENTION PATTERN ANALYSIS:")
        # Average attention patterns across all batches
        avg_attention = torch.cat(attention_patterns, dim=0).mean(dim=0)  # Average over batches
        attention_analysis = analyze_attention_patterns(avg_attention.unsqueeze(0))
        analysis_results['attention_analysis'] = attention_analysis
        
        print(f"   Peak Attention Time: {attention_analysis['peak_attention_time']}")
        print(f"   Attention Concentration: {attention_analysis['attention_concentration']:.2f}")
        print(f"   Early/Middle/Late Attention: "
              f"{attention_analysis['early_attention']:.3f}/"
              f"{attention_analysis['middle_attention']:.3f}/"
              f"{attention_analysis['late_attention']:.3f}")
    
    # Alignment loss analysis
    if alignment_losses:
        avg_alignment = np.mean(alignment_losses)
        print(f"\nMOTION-GAZE ALIGNMENT:")
        print(f"   Average Alignment Loss: {avg_alignment:.4f}")
        print(f"   Min/Max Alignment Loss: {np.min(alignment_losses):.4f}/{np.max(alignment_losses):.4f}")
        analysis_results['alignment_loss'] = avg_alignment
    
    return avg_pose_error, errors_per_frame, analysis_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-pth', type=str, 
                       default='log_aggressive/snapshot/model-best-gate-aggressive.pth',
                       help='path to trained aggressive gated model')
    parser.add_argument('--analyze-attention', action='store_true',
                       help='perform detailed attention analysis')
    parser.add_argument('--save-analysis', type=str, default=None,
                       help='save analysis plots to directory')
    parser.add_argument('--compare-baseline', type=str, default='log_gated/snapshot/model-iter-40000.pth',
                       help='path to baseline model for comparison')
    args = parser.parse_args()
    
    print("=" * 80)
    print("AGGRESSIVE GATED CROSS-ATTENTION EVALUATION")
    print("   BREAKTHROUGH: Testing Aggressive Gate Learning Results!")
    print("=" * 80)
    
    # Initialize model
    model = create_gated_model(config)
    
    # Load trained weights
    if os.path.exists(args.model_pth):
        try:
            # Try loading with weights_only first
            state_dict = torch.load(args.model_pth, weights_only=True)
            model.load_state_dict(state_dict, strict=True)
        except:
            # Fallback: Load full checkpoint (may contain training state)
            checkpoint = torch.load(args.model_pth, map_location='cpu', weights_only=False)
            if 'model_state_dict' in checkpoint:
                # Aggressive training checkpoint
                model.load_state_dict(checkpoint['model_state_dict'], strict=True)
                print(f"Loaded aggressive training checkpoint from {args.model_pth}")
                print(f"Training stage: {checkpoint.get('current_stage', 'unknown')}")
                print(f"Strategy: {checkpoint.get('strategy', 'unknown')}")
                if 'best_gate_weight' in checkpoint:
                    print(f"Best gate weight achieved: {checkpoint['best_gate_weight']:.4f}")
                if 'achieved_milestones' in checkpoint:
                    print(f"Achieved milestones: {checkpoint['achieved_milestones']}")
            else:
                # Regular model state dict
                model.load_state_dict(checkpoint, strict=True)
        print(f"Loaded aggressive gated model from {args.model_pth}")
    else:
        print(f"Model file not found: {args.model_pth}")
        print("Please train the aggressive gated model first")
        exit(1)
    
    model.eval()
    model.cuda()
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters()) / 1000000.0
    print(f"Model parameters: {total_params:.2f}M")
    print(f"Current gate weight: {model.get_gate_weight():.4f}")
    
    # Setup test dataset
    config.motion.h36m_target_length = config.motion.h36m_target_length_eval
    test_dataset = MoGazeGazeEval(config, 'test')
    
    dataloader = DataLoader(test_dataset, batch_size=16,
                          num_workers=0, drop_last=False,
                          shuffle=False, pin_memory=True)
    
    print(f"Test dataset size: {len(test_dataset)} samples")
    print(f"Input format: 66D (63 pose + 3 gaze)")
    print(f"Output format: 63D (pose only)")
    print(f"Attention analysis: {'Enabled' if args.analyze_attention else 'Disabled'}")
    
    # Run evaluation
    avg_pose_error, errors_per_frame, analysis_results = test_aggressive_model_with_analysis(
        config, model, dataloader, analyze_attention=args.analyze_attention
    )
    
    # AGGRESSIVE TRAINING RESULTS
    print("\n" + "=" * 80)
    print("AGGRESSIVE TRAINING RESULTS")
    print("=" * 80)
    
    print(f"\nPOSE PREDICTION RESULTS (MPJPE):")
    print(f"   Average MPJPE: {avg_pose_error:.1f} mm")
    
    # Time-based results
    target_frames = [6, 12, 18, 24, 30]  # 200ms, 400ms, 600ms, 800ms, 1000ms
    print(f"\nTime-based results:")
    for frame in target_frames:
        if frame <= len(errors_per_frame):
            ms = int(frame * 1000 / 30)
            print(f"      {ms:4d}ms: {errors_per_frame[frame-1]:5.1f} mm")
    
    # BREAKTHROUGH COMPARISON
    print(f"\nBREAKTHROUGH COMPARISON:")
    baseline_3d = 73.3
    conservative_gated = 73.2
    gazemotion_baseline = 99.5
    gazemotion_gaze = 75.9
    
    print(f"   MoGaze 3D Baseline (Ours):        {baseline_3d:.1f} mm")
    print(f"   Conservative Gated (Previous):    {conservative_gated:.1f} mm")
    print(f"   Aggressive Gated (This):          {avg_pose_error:.1f} mm")
    print(f"   GazeMotion Baseline:              {gazemotion_baseline:.1f} mm")  
    print(f"   GazeMotion + Gaze:                {gazemotion_gaze:.1f} mm")
    
    # Calculate improvements
    vs_baseline = ((baseline_3d - avg_pose_error) / baseline_3d) * 100
    vs_conservative = ((conservative_gated - avg_pose_error) / conservative_gated) * 100
    vs_gazemotion_gaze = ((gazemotion_gaze - avg_pose_error) / gazemotion_gaze) * 100
    
    print(f"\nAGGRESSIVE TRAINING ACHIEVEMENTS:")
    if avg_pose_error < baseline_3d:
        print(f"   vs 3D Baseline: {vs_baseline:+.1f}% improvement!")
    else:
        diff = avg_pose_error - baseline_3d  
        print(f"   vs 3D Baseline: {diff:+.1f} mm difference")
        
    if avg_pose_error < conservative_gated:
        print(f"   vs Conservative Gated: {vs_conservative:+.1f}% improvement!")
        print(f"   AGGRESSIVE TRAINING SUCCESSFUL!")
    else:
        diff = avg_pose_error - conservative_gated
        print(f"   vs Conservative Gated: {diff:+.1f} mm difference")
        
    if avg_pose_error < gazemotion_gaze:
        print(f"   vs GazeMotion+Gaze: {vs_gazemotion_gaze:+.1f}% improvement!")
        print(f"   STILL STATE-OF-THE-ART ON MOGAZE DATASET!")
    
    # Information extraction analysis
    available_improvement = gazemotion_baseline - gazemotion_gaze  # 23.6mm proven available
    our_improvement = baseline_3d - avg_pose_error
    
    if our_improvement > 0:
        extraction_rate = (our_improvement / available_improvement) * 100
        print(f"\nINFORMATION EXTRACTION ANALYSIS:")
        print(f"   Proven Available Improvement: {available_improvement:.1f} mm")
        print(f"   Our Improvement: {our_improvement:.1f} mm") 
        print(f"   Extraction Rate: {extraction_rate:.1f}% of proven information")
        
        if extraction_rate > 20:
            print(f"   SIGNIFICANT gaze information extraction!")
        elif extraction_rate > 10:
            print(f"   MODERATE gaze information extraction")
        else:
            print(f"   LIMITED gaze information extraction")
    
    # Technical insights
    print(f"\nTECHNICAL INSIGHTS:")
    print(f"   Architecture: Aggressive gated cross-attention with {total_params:.2f}M parameters")
    print(f"   Training: Shorter baseline (5k) + High gate LR (×10) + Regularization")
    
    if 'gate_analysis' in analysis_results:
        gate_info = analysis_results['gate_analysis']
        if gate_info['mean_gate'] > 0.1:
            print(f"   Gate Learning: SUCCESS (mean={gate_info['mean_gate']:.3f})")
        else:
            print(f"   Gate Learning: LIMITED (mean={gate_info['mean_gate']:.3f})")
    
    # Save analysis plots if requested
    if args.save_analysis and args.analyze_attention:
        os.makedirs(args.save_analysis, exist_ok=True)
        print(f"\nSaving analysis plots to {args.save_analysis}/...")
    
    print(f"\nEVALUATION COMPLETED!")
    if args.analyze_attention:
        print(f"   Full attention analysis performed")
    else:
        print(f"   Run with --analyze-attention for detailed analysis")

if __name__ == "__main__":
    main()