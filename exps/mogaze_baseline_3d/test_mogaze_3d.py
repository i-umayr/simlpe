import argparse
import os, sys
import numpy as np
from mogaze_config_3d import config
from model import siMLPe as Model
from mogaze_dataset_3d import MoGaze3DEval
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

def test_mogaze_3d_direct_mpjpe(config, model, dataloader):
    """
    Test function with direct MPJPE calculation on 3D positions
    No conversion needed - both training and evaluation use 3D positions
    """
    model.eval()
    
    # DCT matrices
    dct_m, idct_m = get_dct_matrix(config.motion.h36m_input_length_dct)
    dct_m = torch.tensor(dct_m).float().cuda().unsqueeze(0)
    idct_m = torch.tensor(idct_m).float().cuda().unsqueeze(0)
    
    total_error = 0.0
    num_samples = 0
    errors_per_frame = []
    
    print("🔄 Direct MPJPE evaluation on 3D positions (no conversion needed)...")
    
    with torch.no_grad():
        for batch_idx, (motion_input, motion_target) in enumerate(dataloader):
            # Convert numpy arrays to tensors if needed
            if isinstance(motion_input, np.ndarray):
                motion_input = torch.tensor(motion_input).float()
            if isinstance(motion_target, np.ndarray):
                motion_target = torch.tensor(motion_target).float()
                
            motion_input = motion_input.cuda()
            motion_target = motion_target.cuda()
            
            b, n, c = motion_input.shape
            num_samples += b
            
            # Auto-regressive prediction (same as H36M approach)
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
                
                output = model(motion_input_)
                output = torch.matmul(idct_m[:, :config.motion.h36m_input_length, :], output)[:, :step, :]
                
                if config.deriv_output:
                    output = output + current_input[:, -1:, :].repeat(1, step, 1)
                
                output = output.reshape(-1, 63)  # 21 joints × 3 positions
                output = output.reshape(b, step, -1)
                outputs.append(output)
                
                # Update input for next iteration
                current_input = torch.cat([current_input[:, step:], output], axis=1)
            
            # Concatenate outputs and take first 30 frames
            motion_pred = torch.cat(outputs, axis=1)[:, :30]  # [batch, 30, 63]
            
            # Convert to numpy for MPJPE calculation
            motion_pred_np = motion_pred.cpu().numpy()  # [batch, 30, 63]
            motion_target_np = motion_target.cpu().numpy()  # [batch, 30, 63]
            
            # Reshape to [batch, 30, 21, 3] for joint-wise error calculation
            pred_3d = motion_pred_np.reshape(b, 30, 21, 3)  # [batch, 30, 21, 3]
            target_3d = motion_target_np.reshape(b, 30, 21, 3)  # [batch, 30, 21, 3]
            
            # Calculate MPJPE in meters, then convert to millimeters
            error = np.linalg.norm(pred_3d - target_3d, axis=3)  # [batch, 30, 21]
            error_mm = error * 1000  # Convert to millimeters
            
            # Average over joints and batch for each frame
            frame_errors = np.mean(error_mm, axis=(0, 2))  # [30] - error per frame
            total_error += np.sum(error_mm)
            
            if len(errors_per_frame) == 0:
                errors_per_frame = frame_errors
            else:
                errors_per_frame += frame_errors
                
            # Progress indicator
            if batch_idx % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches...")
    
    # Calculate average errors
    avg_error = total_error / (num_samples * 30 * 21)  # 30 frames, 21 joints
    errors_per_frame = errors_per_frame / len(dataloader)
    
    return avg_error, errors_per_frame

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-pth', type=str, 
                       default='log_3d/snapshot/model-iter-40000.pth',
                       help='path to trained 3D position model')
    args = parser.parse_args()
    
    print("=" * 80)
    print("🎯 MOGAZE 3D POSITION DIRECT MPJPE EVALUATION")
    print("=" * 80)
    
    # Initialize model
    model = Model(config)
    
    # Load trained weights
    if os.path.exists(args.model_pth):
        state_dict = torch.load(args.model_pth, weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        print(f"✅ Loaded 3D position model from {args.model_pth}")
    else:
        print(f"❌ Model file not found: {args.model_pth}")
        print("Please train the 3D position model first using train_mogaze_3d.py")
        exit(1)
    
    model.eval()
    model.cuda()
    
    # Setup test dataset
    config.motion.h36m_target_length = config.motion.h36m_target_length_eval
    test_dataset = MoGaze3DEval(config, 'test')
    
    dataloader = DataLoader(test_dataset, batch_size=16,  # Smaller batch for memory
                          num_workers=0, drop_last=False,
                          shuffle=False, pin_memory=True)
    
    print(f"📊 Test dataset size: {len(test_dataset)} samples")
    print(f"🔄 Data format: 3D positions (21 joints × 3 coordinates = 63 dimensions)")
    print(f"🔄 Running direct MPJPE evaluation (no conversion needed)...")
    
    # Run direct MPJPE evaluation
    avg_error, errors_per_frame = test_mogaze_3d_direct_mpjpe(config, model, dataloader)
    
    print("\n" + "=" * 80)
    print("📊 RESULTS - DIRECT MPJPE ON 3D POSITIONS")
    print("=" * 80)
    
    print(f"\n🎯 YOUR siMLPe 3D RESULTS:")
    print(f"Average MPJPE: {avg_error:.1f} mm")
    
    # Time-based results (30 fps = 33.33ms per frame)
    target_frames = [6, 12, 18, 24, 30]  # 200ms, 400ms, 600ms, 800ms, 1000ms
    print(f"\nTime-based results:")
    for frame in target_frames:
        if frame <= len(errors_per_frame):
            ms = int(frame * 1000 / 30)
            print(f"  {ms:4d}ms: {errors_per_frame[frame-1]:5.1f} mm")
    
    print(f"\n📚 GAZEMOTION PAPER BENCHMARKS (for comparison):")
    print(f"siMLPe baseline on MoGaze (Euler→3D conversion):")
    print(f"  200ms: 40.6 mm,  400ms: 72.0 mm,  600ms: 108.8 mm")
    print(f"  800ms: 152.6 mm, 1000ms: 201.0 mm")
    print(f"  Average: 99.5 mm")
    print(f"")
    print(f"GazeMotion method (with gaze, trained on 3D):")
    print(f"  Average: 75.9 mm (23.7% improvement over baseline)")
    
    print(f"\n🏆 PERFORMANCE COMPARISON:")
    gazemotion_baseline = 99.5
    gazemotion_method = 75.9
    
    if avg_error < gazemotion_baseline:
        improvement_baseline = ((gazemotion_baseline - avg_error) / gazemotion_baseline) * 100
        print(f"  ✅ Your siMLPe 3D BEATS GazeMotion baseline by {improvement_baseline:.1f}%!")
        print(f"     ({avg_error:.1f} mm vs {gazemotion_baseline} mm)")
    else:
        difference_baseline = avg_error - gazemotion_baseline
        print(f"  ⚠️  Your siMLPe 3D: {difference_baseline:.1f} mm above baseline")
        print(f"     ({avg_error:.1f} mm vs {gazemotion_baseline} mm)")
    
    if avg_error < gazemotion_method:
        improvement_method = ((gazemotion_method - avg_error) / gazemotion_method) * 100
        print(f"  🚀 Your siMLPe 3D BEATS GazeMotion method by {improvement_method:.1f}%!")
        print(f"     ({avg_error:.1f} mm vs {gazemotion_method} mm)")
    else:
        difference_method = avg_error - gazemotion_method
        print(f"  📈 Room for improvement: {difference_method:.1f} mm gap to GazeMotion")
        print(f"     ({avg_error:.1f} mm vs {gazemotion_method} mm)")
    
    print(f"\n💡 KEY INSIGHTS:")
    print(f"  ✅ Fair comparison: Both models trained on 3D positions")
    print(f"  ✅ No conversion errors: Direct MPJPE calculation")
    print(f"  ✅ Same data representation as GazeMotion paper")
    
    print(f"\n🎯 Evaluation completed!")

if __name__ == "__main__":
    main()