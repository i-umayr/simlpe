"""
Setup script for MoGaze training with siMLPe
Run this to set up the necessary files and directories
"""

import os
import shutil

def setup_mogaze_experiment():
    # Create experiment directory
    exp_dir = "exps/mogaze_baseline"
    os.makedirs(exp_dir, exist_ok=True)
    
    print(f"Created experiment directory: {exp_dir}")
    
    # Copy necessary files from H36M experiment
    source_files = [
        ("exps/baseline_h36m/model.py", f"{exp_dir}/model.py"),
        ("exps/baseline_h36m/mlp.py", f"{exp_dir}/mlp.py"),
        ("lib/utils/logger.py", f"{exp_dir}/logger.py"),
        ("lib/utils/pyt_utils.py", f"{exp_dir}/pyt_utils.py")
    ]
    
    for src, dst in source_files:
        if os.path.exists(src):
            # Create target directory if it doesn't exist
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            print(f"Copied {src} -> {dst}")
        else:
            print(f"Warning: Source file not found: {src}")
    
    # Create log directory
    log_dir = f"{exp_dir}/log"
    os.makedirs(log_dir, exist_ok=True)
    print(f"Created log directory: {log_dir}")
    
    print("\nSetup completed!")
    print(f"\nNext steps:")
    print(f"1. Place the provided files in {exp_dir}/:")
    print(f"   - mogaze_config.py")
    print(f"   - mogaze_dataset.py") 
    print(f"   - train_mogaze.py")
    print(f"   - test_mogaze.py")
    print(f"2. cd {exp_dir}")
    print(f"3. python train_mogaze.py --with-normalization")

if __name__ == "__main__":
    setup_mogaze_experiment()