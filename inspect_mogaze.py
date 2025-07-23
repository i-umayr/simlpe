import h5py
import numpy as np
import os

def inspect_mogaze_dataset(data_path):
    """Inspect MoGaze dataset structure"""
    print("=== MoGaze Dataset Inspector ===")
    
    # List all files in the directory
    files = [f for f in os.listdir(data_path) if f.endswith('.hdf5')]
    print(f"Found {len(files)} HDF5 files:")
    for f in files:
        print(f"  - {f}")
    
    if not files:
        print("No HDF5 files found! Check your data path.")
        return
    
    # Take the first human data file as example
    human_files = [f for f in files if 'human_data' in f]
    if not human_files:
        print("No human_data files found!")
        return
    
    sample_file = human_files[0]
    print(f"\nInspecting: {sample_file}")
    
    with h5py.File(os.path.join(data_path, sample_file), 'r') as f:
        print(f"\nFile structure:")
        def print_structure(name, obj):
            print(f"  {name}: {type(obj).__name__}")
            if isinstance(obj, h5py.Dataset):
                print(f"    Shape: {obj.shape}")
                print(f"    Dtype: {obj.dtype}")
                if hasattr(obj, 'attrs') and len(obj.attrs) > 0:
                    print(f"    Attributes: {dict(obj.attrs)}")
        
        f.visititems(print_structure)
        
        # Try to load some data
        if 'data' in f:
            data = f['data'][:]
            print(f"\nData analysis:")
            print(f"  Total frames: {data.shape[0]}")
            print(f"  Data shape: {data.shape}")
            print(f"  Data type: {data.dtype}")
            
            # If it's joint data, try to figure out the structure
            if len(data.shape) >= 2:
                total_dims = data.shape[1] if len(data.shape) == 2 else np.prod(data.shape[1:])
                possible_joints = total_dims // 3  # Assuming 3D coordinates
                print(f"  Total dimensions: {total_dims}")
                print(f"  Possible number of joints (if 3D): {possible_joints}")
                
                # Show a sample frame
                print(f"\nFirst frame sample (first 30 values):")
                sample_frame = data[0].flatten()[:30]
                print(f"  {sample_frame}")
                
                # Check if data is structured as [frames, joints, 3] or [frames, joints*3]
                if len(data.shape) == 3 and data.shape[2] == 3:
                    print(f"  Data appears to be structured as [frames, {data.shape[1]} joints, 3 coordinates]")
                elif len(data.shape) == 2 and data.shape[1] % 3 == 0:
                    joints = data.shape[1] // 3
                    print(f"  Data appears to be structured as [frames, {joints} joints * 3 coordinates]")
                
                # Try to reshape and show joint positions for first frame
                try:
                    if len(data.shape) == 2 and data.shape[1] % 3 == 0:
                        joints = data.shape[1] // 3
                        reshaped = data[0].reshape(joints, 3)
                        print(f"\nFirst frame joint positions (showing first 5 joints):")
                        for i in range(min(5, joints)):
                            print(f"  Joint {i}: [{reshaped[i, 0]:.3f}, {reshaped[i, 1]:.3f}, {reshaped[i, 2]:.3f}]")
                    elif len(data.shape) == 3:
                        print(f"\nFirst frame joint positions (showing first 5 joints):")
                        for i in range(min(5, data.shape[1])):
                            print(f"  Joint {i}: [{data[0, i, 0]:.3f}, {data[0, i, 1]:.3f}, {data[0, i, 2]:.3f}]")
                except:
                    print("  Could not reshape data to show joint positions")

if __name__ == "__main__":
    # Adjust this path to your MoGaze data location
    mogaze_path = r"D:\siMLPe\data\mogaze"
    
    if os.path.exists(mogaze_path):
        inspect_mogaze_dataset(mogaze_path)
    else:
        print(f"Path not found: {mogaze_path}")
        print("Please check your data path and update the script.")