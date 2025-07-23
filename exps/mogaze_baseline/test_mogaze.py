import argparse
import os, sys
import numpy as np
from mogaze_config import config
from model import siMLPe as Model
from mogaze_dataset import MoGazeEval
import torch
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation as R

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

def euler2xyz(pose_euler):
    """
    Convert the human pose in MoGaze dataset from euler representation to xyz representation.
    Exact same function as used by GazeMotion paper.
    """     
    # names of all the 21 joints
    joint_names = ['base', 'pelvis', 'torso', 'neck', 'head', 'linnerShoulder',  
                         'lShoulder', 'lElbow', 'lWrist', 'rinnerShoulder', 'rShoulder', 
                         'rElbow', 'rWrist', 'lHip', 'lKnee', 'lAnkle', 
                         'lToe', 'rHip', 'rKnee', 'rAnkle', 'rToe']                                                                                   
    joint_ids = {name: idx for idx, name in enumerate(joint_names)}
                         
    # translation of the 20 joints (excluding base), obtained from the mogaze dataset
    joint_trans = np.array([[0, 0, 0.074],
                            [0, 0, 0.201],
                            [0, 0, 0.234], 
                            [0, -0.018, 0.140],
                            [0.036, 0, 0.183], 
                            [0.153, 0, 0],
                            [0.243, 0, 0], 
                            [0.267, -0.002, 0],
                            [-0.036, 0, 0.183], 
                            [-0.153, 0, 0],
                            [-0.243, 0, 0], 
                            [-0.267, -0.002, 0],
                            [0.090, 0, 0], 
                            [0, 0, -0.383],
                            [0, 0, -0.354], 
                            [0, -0.135, -0.059],
                            [-0.090, 0, 0], 
                            [0, 0, -0.383],
                            [0, 0, -0.354], 
                            [0, -0.135, -0.059]])
         
    # parent of every joint
    joint_parent_names = {
                                  # root
                                  'base':           'base',
                                  'pelvis':         'base',                               
                                  'torso':          'pelvis', 
                                  'neck':           'torso', 
                                  'head':           'neck', 
                                  'linnerShoulder': 'torso',
                                  'lShoulder':      'linnerShoulder', 
                                  'lElbow':         'lShoulder', 
                                  'lWrist':         'lElbow', 
                                  'rinnerShoulder': 'torso', 
                                  'rShoulder':      'rinnerShoulder', 
                                  'rElbow':         'rShoulder', 
                                  'rWrist':         'rElbow', 
                                  'lHip':           'base', 
                                  'lKnee':          'lHip', 
                                  'lAnkle':         'lKnee', 
                                  'lToe':           'lAnkle', 
                                  'rHip':           'base', 
                                  'rKnee':          'rHip', 
                                  'rAnkle':         'rKnee', 
                                  'rToe':           'rAnkle'}                               
    # id of joint parent
    joint_parent_ids = [joint_ids[joint_parent_names[child_name]] for child_name in joint_names]
        
    # forward kinematics
    joint_number = len(joint_names)
    pose_xyz = np.zeros((pose_euler.shape[0], joint_number*3))
    for i in range(pose_euler.shape[0]):        
        # xyz position in the world coordinate system
        pose_xyz_tmp = np.zeros((joint_number, 3))
        pose_xyz_tmp[0] = [pose_euler[i][0], pose_euler[i][1], pose_euler[i][2]]                        
        pose_rot_mat = np.zeros((joint_number, 3, 3))
        for j in range(joint_number):
            rot = np.array([pose_euler[i][(j+1)*3], pose_euler[i][(j+1)*3 + 1], pose_euler[i][(j+1)*3 + 2]])
            pose_rot_mat[j] = R.from_euler('XYZ', rot).as_matrix()
                          
        for j in range(1, joint_number):
            pose_rot_mat_parent = pose_rot_mat[joint_parent_ids[j]]
            pose_xyz_tmp[j] = np.matmul(pose_rot_mat_parent, joint_trans[j-1]) + pose_xyz_tmp[joint_parent_ids[j]]
            pose_rot_mat[j] = np.matmul(pose_rot_mat_parent, pose_rot_mat[j])
        
        pose_xyz[i] = pose_xyz_tmp.reshape(joint_number*3)
    return pose_xyz

def test_mogaze_proper_mpjpe(config, model, dataloader):
    """
    Test function with proper forward kinematics conversion to get real MPJPE in millimeters
    """
    model.eval()
    
    # DCT matrices
    dct_m, idct_m = get_dct_matrix(config.motion.h36m_input_length_dct)
    dct_m = torch.tensor(dct_m).float().cuda().unsqueeze(0)
    idct_m = torch.tensor(idct_m).float().cuda().unsqueeze(0)
    
    total_error = 0.0
    num_samples = 0
    errors_per_frame = []
    
    print("🔄 Converting Euler angle predictions to 3D positions using forward kinematics...")
    
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
            
            # Auto-regressive prediction (same as your current approach)
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
                
                output = output.reshape(-1, 66)
                output = output.reshape(b, step, -1)
                outputs.append(output)
                
                # Update input for next iteration
                current_input = torch.cat([current_input[:, step:], output], axis=1)
            
            # Concatenate outputs and take first 30 frames
            motion_pred = torch.cat(outputs, axis=1)[:, :30]  # [batch, 30, 66]
            
            # Convert to numpy for forward kinematics
            motion_pred_np = motion_pred.cpu().numpy()  # [batch, 30, 66]
            motion_target_np = motion_target.cpu().numpy()  # [batch, 30, 66]
            
            # Apply forward kinematics to convert Euler angles to 3D positions
            batch_pred_3d = []
            batch_target_3d = []
            
            for sample_idx in range(b):
                # Convert predictions: [30, 66] -> [30, 21*3] -> [30, 21, 3]
                pred_sample = motion_pred_np[sample_idx]  # [30, 66]
                pred_3d = euler2xyz(pred_sample)  # [30, 21*3]
                pred_3d = pred_3d.reshape(30, 21, 3)  # [30, 21, 3]
                batch_pred_3d.append(pred_3d)
                
                # Convert targets: [30, 66] -> [30, 21*3] -> [30, 21, 3]
                target_sample = motion_target_np[sample_idx]  # [30, 66]
                target_3d = euler2xyz(target_sample)  # [30, 21*3]
                target_3d = target_3d.reshape(30, 21, 3)  # [30, 21, 3]
                batch_target_3d.append(target_3d)
            
            # Convert back to tensors
            batch_pred_3d = np.array(batch_pred_3d)  # [batch, 30, 21, 3]
            batch_target_3d = np.array(batch_target_3d)  # [batch, 30, 21, 3]
            
            # Calculate MPJPE in meters, then convert to millimeters
            error = np.linalg.norm(batch_pred_3d - batch_target_3d, axis=3)  # [batch, 30, 21]
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
                       default='log/snapshot/model-iter-40000.pth',
                       help='path to trained model')
    args = parser.parse_args()
    
    print("=" * 80)
    print("🎯 MOGAZE PROPER MPJPE EVALUATION")
    print("=" * 80)
    
    # Initialize model
    model = Model(config)
    
    # Load trained weights
    if os.path.exists(args.model_pth):
        state_dict = torch.load(args.model_pth, weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        print(f"✅ Loaded model from {args.model_pth}")
    else:
        print(f"❌ Model file not found: {args.model_pth}")
        exit(1)
    
    model.eval()
    model.cuda()
    
    # Setup test dataset
    config.motion.h36m_target_length = config.motion.h36m_target_length_eval
    test_dataset = MoGazeEval(config, 'test')
    
    dataloader = DataLoader(test_dataset, batch_size=16,  # Smaller batch for memory
                          num_workers=0, drop_last=False,
                          shuffle=False, pin_memory=True)
    
    print(f"📊 Test dataset size: {len(test_dataset)} samples")
    print(f"🔄 Running evaluation with proper forward kinematics...")
    
    # Run proper MPJPE evaluation
    avg_error, errors_per_frame = test_mogaze_proper_mpjpe(config, model, dataloader)
    
    print("\n" + "=" * 80)
    print("📊 RESULTS - PROPER MPJPE IN MILLIMETERS")
    print("=" * 80)
    
    print(f"\n🎯 YOUR siMLPe RESULTS:")
    print(f"Average MPJPE: {avg_error:.1f} mm")
    
    # Time-based results (30 fps = 33.33ms per frame)
    target_frames = [6, 12, 18, 24, 30]  # 200ms, 400ms, 600ms, 800ms, 1000ms
    print(f"\nTime-based results:")
    for frame in target_frames:
        if frame <= len(errors_per_frame):
            ms = int(frame * 1000 / 30)
            print(f"  {ms:4d}ms: {errors_per_frame[frame-1]:5.1f} mm")
    
    print(f"\n📚 GAZEMOTION PAPER BENCHMARKS:")
    print(f"siMLPe baseline (no gaze):")
    print(f"  200ms: 40.6 mm,  400ms: 72.0 mm,  600ms: 108.8 mm")
    print(f"  800ms: 152.6 mm, 1000ms: 201.0 mm")
    print(f"  Average: 99.5 mm")
    print(f"")
    print(f"GazeMotion method (with gaze):")
    print(f"  Average: 75.9 mm (7.4% improvement)")
    
    print(f"\n🏆 PERFORMANCE COMPARISON:")
    gazemotion_baseline = 99.5
    gazemotion_method = 75.9
    
    if avg_error < gazemotion_baseline:
        improvement_baseline = ((gazemotion_baseline - avg_error) / gazemotion_baseline) * 100
        print(f"  ✅ Your siMLPe BEATS baseline by {improvement_baseline:.1f}%!")
        print(f"     ({avg_error:.1f} mm vs {gazemotion_baseline} mm)")
    else:
        difference_baseline = avg_error - gazemotion_baseline
        print(f"  ⚠️  Your siMLPe: {difference_baseline:.1f} mm above baseline")
        print(f"     ({avg_error:.1f} mm vs {gazemotion_baseline} mm)")
    
    if avg_error < gazemotion_method:
        improvement_method = ((gazemotion_method - avg_error) / gazemotion_method) * 100
        print(f"  🚀 Your siMLPe BEATS GazeMotion by {improvement_method:.1f}%!")
        print(f"     ({avg_error:.1f} mm vs {gazemotion_method} mm)")
    else:
        difference_method = avg_error - gazemotion_method
        print(f"  📈 Room for improvement: {difference_method:.1f} mm gap to GazeMotion")
        print(f"     ({avg_error:.1f} mm vs {gazemotion_method} mm)")
    
    print(f"\n💡 NEXT STEPS:")
    if avg_error > gazemotion_baseline:
        print(f"  1. Current approach needs improvement")
        print(f"  2. Consider retraining on 3D positions (like GazeMotion)")
        print(f"  3. Investigate training hyperparameters")
    else:
        print(f"  1. ✅ Excellent results! Consider publishing")
        print(f"  2. Compare with gaze-based improvements")
        print(f"  3. Ablation studies on architecture choices")
    
    print(f"\n🎯 Evaluation completed!")

if __name__ == "__main__":
    main()