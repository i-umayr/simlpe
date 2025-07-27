#!/usr/bin/env python3
"""
Quick check of MoGaze data files - run this first to see what we have
"""

import os
import numpy as np

def quick_check():
    data_dir = "D:/siMLPe/data/mogaze_3d/"
    
    # Also check if original MoGaze files exist for potential reprocessing
    original_dir = "D:/siMLPe/data/mogaze/"
    
    print("🔍 Quick MoGaze Data Check")
    print("=" * 40)
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        print(f"❌ Directory not found: {data_dir}")
        # Check if we need to run preprocessing
        if os.path.exists(original_dir):
            print(f"✅ Found original MoGaze data at: {original_dir}")
            print(f"💡 Run your preprocessing script to generate 3D + gaze data")
        return
    
    # List files
    files = os.listdir(data_dir)
    print(f"📁 Directory: {data_dir}")
    print(f"📊 Total files: {len(files)}")
    
    # Count file types
    pose_count = sum(1 for f in files if "_pose_3d.npy" in f)
    gaze_count = sum(1 for f in files if "_gaze.npy" in f)
    euler_count = sum(1 for f in files if "_pose_euler.npy" in f)
    meta_count = sum(1 for f in files if "_metadata.npy" in f)
    
    print(f"\n📋 File breakdown:")
    print(f"  Pose 3D files: {pose_count}")
    print(f"  Gaze files: {gaze_count}")
    print(f"  Euler files: {euler_count}")
    print(f"  Metadata files: {meta_count}")
    
    # Show first 10 files
    print(f"\n📄 First 10 files:")
    for i, filename in enumerate(files[:10]):
        print(f"  {i+1:2d}. {filename}")
    
    if len(files) > 10:
        print(f"  ... and {len(files) - 10} more")
    
    # Try to load one pose and one gaze file
    print(f"\n🔍 Sample file inspection:")
    
    pose_files = [f for f in files if "_pose_3d.npy" in f]
    gaze_files = [f for f in files if "_gaze.npy" in f]
    
    if pose_files:
        try:
            sample_pose = pose_files[0]
            pose_data = np.load(os.path.join(data_dir, sample_pose))
            print(f"  📊 {sample_pose}:")
            print(f"    Shape: {pose_data.shape}")
            print(f"    Type: {pose_data.dtype}")
            print(f"    Range: [{pose_data.min():.3f}, {pose_data.max():.3f}]")
        except Exception as e:
            print(f"  ❌ Error loading pose file: {e}")
    else:
        print(f"  ❌ No pose files found")
    
    if gaze_files:
        try:
            sample_gaze = gaze_files[0]
            gaze_data = np.load(os.path.join(data_dir, sample_gaze))
            print(f"  👁️ {sample_gaze}:")
            print(f"    Shape: {gaze_data.shape}")
            print(f"    Type: {gaze_data.dtype}")
            print(f"    Range: [{gaze_data.min():.3f}, {gaze_data.max():.3f}]")
            
            # Check if unit vectors
            if len(gaze_data.shape) == 2 and gaze_data.shape[1] == 3:
                magnitudes = np.linalg.norm(gaze_data, axis=1)
                print(f"    Magnitude: mean={np.mean(magnitudes):.3f}, std={np.std(magnitudes):.3f}")
            
        except Exception as e:
            print(f"  ❌ Error loading gaze file: {e}")
    else:
        print(f"  ❌ No gaze files found")
    
    # Check if we can find matching pairs
    if pose_files and gaze_files:
        print(f"\n🔗 Checking file pairs:")
        matches = 0
        for pose_file in pose_files[:5]:  # Check first 5
            gaze_file = pose_file.replace("_pose_3d.npy", "_gaze.npy")
            if gaze_file in gaze_files:
                matches += 1
                print(f"  ✅ {pose_file} <-> {gaze_file}")
            else:
                print(f"  ❌ {pose_file} (no matching gaze file)")
        
        print(f"  Found {matches} matching pairs out of {min(5, len(pose_files))} checked")
    
    # Final assessment
    ready = pose_count > 0 and gaze_count > 0 and pose_count == gaze_count
    print(f"\n🎯 Status: {'✅ READY for gaze integration' if ready else '⚠️ NEEDS ATTENTION'}")
    
    if ready:
        print(f"✨ You have {pose_count} pose-gaze pairs ready for training!")
    else:
        print(f"⚠️ Missing gaze data - may need to run preprocessing")

if __name__ == "__main__":
    quick_check()