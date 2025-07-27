import argparse
import os, sys
import numpy as np
from mogaze_config_gaze import config
from model import siMLPe as Model
from mogaze_dataset_gaze import MoGazeGazeEval
import torch
from torch.utils.data import DataLoader

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

def test_mogaze_gaze_direct_mpjpe(config, model, dataloader):
    """
    CORRECTED: Test function for gaze integration model
    - Input: 66D (63 pose + 3 gaze)
    - Model Output: 63D (pose only) 
    - Evaluation: MPJPE on pose only (fair comparison with baseline)
    """
    model.eval()
    
    # DCT matrices
    dct_m, idct_m = get_dct_matrix(config.motion.h36m_input_length_dct)
    dct_m = torch.tensor(dct_m).float().cuda().unsqueeze(0)
    idct_m = torch.tensor(idct_m).float().cuda().unsqueeze(0)
    
    total_error = 0.0
    num_samples = 0
    errors_per_frame = []
    
    print("Direct MPJPE evaluation: Gaze as input, pose-only predictions...")
    
    with torch.no_grad():
        for batch_idx, (motion_input, motion_target) in enumerate(dataloader):
            # Convert numpy arrays to tensors if needed
            if isinstance(motion_input, np.ndarray):
                motion_input = torch.tensor(motion_input).float()
            if isinstance(motion_target, np.ndarray):
                motion_target = torch.tensor(motion_target).float()
                
            motion_input = motion_input.cuda()  # [batch, 50, 66] - pose+gaze input
            motion_target = motion_target.cuda()  # [batch, 30, 66] - pose+gaze target
            
            b, n, c = motion_input.shape
            num_samples += b
            
            # Auto-regressive prediction (same as H36M/baseline approach)
            outputs = []
            step = config.motion.h36m_target_length_train  # 10 frames per step
            
            if step == 30:
                num_step = 1
            else:
                num_step = 30 // step + 1
            
            current_input = motion_input.clone()
            
            for idx in range(num_step):
                # Forward pass with exact same logic as training
                if config.deriv_input:
                    motion_input_ = current_input.clone()
                    motion_input_ = torch.matmul(dct_m[:, :, :config.motion.h36m_input_length], motion_input_)
                else:
                    motion_input_ = current_input.clone()
                
                # CORRECTED: Model outputs 63D poses only (not 66D)
                output = model(motion_input_)  # [batch, 10, 63] - pose only
                output = torch.matmul(idct_m[:, :config.motion.h36m_input_length, :], output)[:, :step, :]
                
                if config.deriv_output:
                    # CORRECTED: Add residual from pose component only
                    pose_offset = current_input[:, -1:, :63]  # Extract pose from input
                    output = output + pose_offset
                
                # CORRECTED: Pad output back to 66D for next iteration input
                # Model predicts pose (63D), but input needs pose+gaze (66D)
                batch_size, seq_len, _ = output.shape
                
                # Keep gaze from the last frame and extend it
                last_gaze = current_input[:, -1:, 63:66]  # [batch, 1, 3]
                extended_gaze = last_gaze.repeat(1, seq_len, 1)  # [batch, seq_len, 3]
                
                # Combine predicted pose + extended gaze for next input
                output_with_gaze = torch.cat([output, extended_gaze], dim=2)  # [batch, seq_len, 66]
                outputs.append(output)  # Store pose-only predictions for evaluation
                
                # Update input for next iteration (66D: predicted pose + extended gaze)
                current_input = torch.cat([current_input[:, step:], output_with_gaze], axis=1)
            
            # Concatenate pose-only outputs and take first 30 frames
            motion_pred = torch.cat(outputs, axis=1)[:, :30]  # [batch, 30, 63] - pose only
            
            # Convert to numpy for MPJPE calculation
            motion_pred_np = motion_pred.cpu().numpy()  # [batch, 30, 63]
            motion_target_np = motion_target.cpu().numpy()  # [batch, 30, 66]
            
            # Extract pose component from target for fair comparison
            target_pose = motion_target_np[:, :, :63]  # [batch, 30, 63] - pose only
            
            # Reshape to [batch, 30, 21, 3] for joint-wise error calculation
            pred_pose_3d = motion_pred_np.reshape(b, 30, 21, 3)  # [batch, 30, 21, 3]
            target_pose_3d = target_pose.reshape(b, 30, 21, 3)  # [batch, 30, 21, 3]
            
            # Calculate MPJPE on pose component in meters, then convert to millimeters
            pose_error = np.linalg.norm(pred_pose_3d - target_pose_3d, axis=3)  # [batch, 30, 21]
            pose_error_mm = pose_error * 1000  # Convert to millimeters
            
            # Average over joints and batch for each frame
            frame_errors = np.mean(pose_error_mm, axis=(0, 2))  # [30] - pose error per frame
            total_error += np.sum(pose_error_mm)
            
            if len(errors_per_frame) == 0:
                errors_per_frame = frame_errors
            else:
                errors_per_frame += frame_errors
                
            # Progress indicator
            if batch_idx % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches...")
    
    # Calculate average errors
    avg_pose_error = total_error / (num_samples * 30 * 21)  # 30 frames, 21 joints
    errors_per_frame = errors_per_frame / len(dataloader)
    
    return avg_pose_error, errors_per_frame

def test(config, model, dataloader):
    """Main test function returning results in same format as baseline"""
    avg_pose_error, errors_per_frame = test_mogaze_gaze_direct_mpjpe(config, model, dataloader)
    
    # Return pose errors in same time format as baseline for comparison
    # Target frames: 200ms, 400ms, 600ms, 800ms, 1000ms at 30fps
    target_frames = [6, 12, 18, 24, 30]  # 200ms, 400ms, 600ms, 800ms, 1000ms
    
    results = []
    for frame in target_frames:
        if frame <= len(errors_per_frame):
            results.append(round(errors_per_frame[frame-1], 1))
    
    # Add average as first element for logging
    results.insert(0, round(avg_pose_error, 1))
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-pth', type=str, 
                       default='log_gaze/snapshot/model-iter-40000.pth',
                       help='path to trained gaze integration model')
    args = parser.parse_args()
    
    print("=" * 80)
    print("MOGAZE GAZE INTEGRATION EVALUATION (CORRECTED)")
    print("=" * 80)
    
    # Initialize model
    model = Model(config)
    
    # Load trained weights
    if os.path.exists(args.model_pth):
        state_dict = torch.load(args.model_pth, weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        print(f"Loaded gaze integration model from {args.model_pth}")
    else:
        print(f"Model file not found: {args.model_pth}")
        print("Please train the gaze integration model first using train_mogaze_gaze.py")
        exit(1)
    
    model.eval()
    model.cuda()
    
    # Setup test dataset
    config.motion.h36m_target_length = config.motion.h36m_target_length_eval
    test_dataset = MoGazeGazeEval(config, 'test')
    
    dataloader = DataLoader(test_dataset, batch_size=16,  # Smaller batch for memory
                          num_workers=0, drop_last=False,
                          shuffle=False, pin_memory=True)
    
    print(f"Test dataset size: {len(test_dataset)} samples")
    print(f"Data format: Input 66D (63 pose + 3 gaze), Output 63D (pose only)")
    print(f"Running evaluation: Gaze as input for better pose prediction...")
    
    # Run evaluation
    avg_pose_error, errors_per_frame = test_mogaze_gaze_direct_mpjpe(config, model, dataloader)
    
    print("\n" + "=" * 80)
    print("GAZE INTEGRATION RESULTS")
    print("=" * 80)
    
    print(f"\nPOSE PREDICTION RESULTS (MPJPE):")
    print(f"Average MPJPE: {avg_pose_error:.1f} mm")
    
    # Time-based results (30 fps = 33.33ms per frame)
    target_frames = [6, 12, 18, 24, 30]  # 200ms, 400ms, 600ms, 800ms, 1000ms
    print(f"\nTime-based pose results:")
    for frame in target_frames:
        if frame <= len(errors_per_frame):
            ms = int(frame * 1000 / 30)
            print(f"  {ms:4d}ms: {errors_per_frame[frame-1]:5.1f} mm")
    
    print(f"\nBASELINE COMPARISON:")
    baseline_3d = 73.3
    print(f"3D Pose Only Baseline: {baseline_3d} mm")
    print(f"Gaze Integration (This): {avg_pose_error:.1f} mm")
    
    if avg_pose_error < baseline_3d:
        improvement = ((baseline_3d - avg_pose_error) / baseline_3d) * 100
        print(f"IMPROVEMENT: {improvement:.1f}% better than 3D baseline!")
        print(f"   ({avg_pose_error:.1f} mm vs {baseline_3d} mm)")
    else:
        difference = avg_pose_error - baseline_3d
        print(f"Performance: {difference:.1f} mm above baseline")
        print(f"   ({avg_pose_error:.1f} mm vs {baseline_3d} mm)")
    
    print(f"\nKEY INSIGHTS:")
    print(f"  Model uses gaze as input to predict better poses")
    print(f"  Output is pose-only (63D) for fair comparison with baseline")
    print(f"  Same evaluation methodology as 73.3mm baseline")
    
    print(f"\nEvaluation completed!")

if __name__ == "__main__":
    main()