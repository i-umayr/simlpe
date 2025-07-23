import os
import glob
import numpy as np
import h5py
from tqdm import tqdm
import torch
import torch.utils.data as data

class MoGazeDataset(data.Dataset):
    def __init__(self, config, split_name, data_aug=False):
        super(MoGazeDataset, self).__init__()
        self._split_name = split_name
        self.data_aug = data_aug
        
        self._mogaze_anno_dir = config.mogaze_anno_dir
        
        # FIXED: Use the correct config attribute names
        self.mogaze_motion_input_length = config.motion.mogaze_input_length  # 50 frames
        
        # FIXED: Use train vs eval target length based on split
        if split_name == 'train':
            self.mogaze_motion_target_length = config.motion.mogaze_target_length_train  # 10 frames
        else:
            self.mogaze_motion_target_length = config.motion.mogaze_target_length_eval   # 30 frames
        
        # MoGaze has 22 joints with 3 rotation values each = 66 dimensions
        self.motion_dim = config.motion.dim  # 66 for MoGaze (22 joints × 3 rotations)
        
        self.shift_step = config.shift_step
        
        # Get participant splits (leave-one-person-out)
        self.participant_splits = self._get_participant_splits()
        self._mogaze_files = self._get_mogaze_files()
        
        self._collect_all()
        self._file_length = len(self.data_idx)
    
    def __len__(self):
        return self._file_length
    
    def _get_participant_splits(self):
        """Define participant splits for cross-validation"""
        # Based on MoGaze dataset structure from inspection
        # Participants with both human and gaze data
        participants = ['p1_1', 'p1_2', 'p2_1', 'p4_1', 'p5_1', 'p6_1', 'p6_2', 'p7_1', 'p7_2', 'p7_3']
        
        # For leave-one-person-out, we'll do a simple split for now
        # You can modify this to implement proper cross-validation
        if self._split_name == 'train':
            return participants[:-2]  # All but last 2 participants
        else:
            return participants[-2:]  # Last 2 participants for testing
    
    def _get_mogaze_files(self):
        """Get MoGaze motion files based on participant split"""
        mogaze_files = []
        
        for participant in self.participant_splits:
            file_pattern = os.path.join(self._mogaze_anno_dir, f"{participant}_human_data.hdf5")
            if os.path.exists(file_pattern):
                mogaze_files.append(file_pattern)
            else:
                print(f"Warning: File not found: {file_pattern}")
        
        print(f"Found {len(mogaze_files)} MoGaze files for {self._split_name} split")
        return mogaze_files
    
    def _load_mogaze_file(self, file_path):
        """Load a single MoGaze human motion file"""
        try:
            with h5py.File(file_path, 'r') as f:
                # Load the motion data (Euler angles/rotations)
                if 'data' in f:
                    motion_data = f['data'][:]
                else:
                    raise KeyError("'data' key not found in HDF5 file")
                
                # Downsample from 120Hz to 30Hz (every 4th frame)
                # Following GazeMotion preprocessing
                motion_data = motion_data[::4]  
                
                # MoGaze data is already in the correct format: [frames, 66]
                # where 66 = 22 joints × 3 rotation values
                frames, total_dims = motion_data.shape
                
                if total_dims != 66:
                    raise ValueError(f"Expected 66 dimensions (22 joints × 3), got {total_dims}")
                
                print(f"Loaded {file_path}: {motion_data.shape[0]} frames, 22 joints (66 dimensions)")
                return motion_data
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def _collect_all(self):
        """Collect all valid motion sequences"""
        self.mogaze_seqs = []
        self.data_idx = []
        idx = 0
        
        for file_path in tqdm(self._mogaze_files, desc="Loading MoGaze files"):
            motion_data = self._load_mogaze_file(file_path)
            
            if motion_data is None:
                continue
            
            N = motion_data.shape[0]
            
            # Skip sequences that are too short
            min_length = self.mogaze_motion_input_length + self.mogaze_motion_target_length
            if N < min_length:
                print(f"Skipping short sequence: {N} < {min_length}")
                continue
            
            # Data is already in the correct format [frames, 66] for siMLPe
            # No need to reshape - this matches H36M format
            self.mogaze_seqs.append(motion_data)
            
            # Create valid frame indices
            valid_frames = np.arange(0, N - min_length + 1, self.shift_step)
            self.data_idx.extend(zip([idx] * len(valid_frames), valid_frames.tolist()))
            idx += 1
        
        print(f"Collected {len(self.mogaze_seqs)} sequences with {len(self.data_idx)} samples")
    
    def __getitem__(self, index):
        idx, start_frame = self.data_idx[index]
        
        # Get the motion sequence (input + target frames)
        total_frames = self.mogaze_motion_input_length + self.mogaze_motion_target_length
        end_frame = start_frame + total_frames
        motion = self.mogaze_seqs[idx][start_frame:end_frame]
        
        # Apply data augmentation if enabled
        if self.data_aug:
            if torch.rand(1)[0] > 0.5:
                # Reverse the sequence (front-back flip)
                idx_reverse = [i for i in range(motion.shape[0]-1, -1, -1)]
                motion = motion[idx_reverse]
        
        # Split into input and target frames
        # Note: MoGaze data is in Euler angles, not positions, so we keep original scale
        mogaze_motion_input = motion[:self.mogaze_motion_input_length]
        mogaze_motion_target = motion[self.mogaze_motion_input_length:]
        
        return torch.tensor(mogaze_motion_input).float(), torch.tensor(mogaze_motion_target).float()

class MoGazeEval(data.Dataset):
    """Evaluation dataset for MoGaze - similar to H36MEval"""
    def __init__(self, config, split_name):
        super(MoGazeEval, self).__init__()
        self._split_name = split_name
        self._mogaze_anno_dir = config.mogaze_anno_dir
        
        self.mogaze_motion_input_length = config.motion.mogaze_input_length
        
        # FIXED: Use the correct config attribute name
        self.mogaze_motion_target_length = config.motion.mogaze_target_length_eval  # Should be 30 for eval
        
        self.motion_dim = config.motion.dim
        self.shift_step = config.shift_step
        
        # For evaluation, we might want to use all participants or specific test set
        self.test_participants = ['p6_1', 'p7_1']  # Example test participants
        self._mogaze_files = self._get_test_files()
        self._collect_all()
        self._file_length = len(self.data_idx)
    
    def __len__(self):
        return self._file_length
    
    def _get_test_files(self):
        """Get test files for evaluation"""
        test_files = []
        for participant in self.test_participants:
            file_pattern = os.path.join(self._mogaze_anno_dir, f"{participant}_human_data.hdf5")
            if os.path.exists(file_pattern):
                test_files.append(file_pattern)
        return test_files
    
    def _collect_all(self):
        """Collect evaluation sequences"""
        self.mogaze_seqs = []
        self.data_idx = []
        idx = 0
        
        for file_path in self._mogaze_files:
            # Load similar to training dataset
            with h5py.File(file_path, 'r') as f:
                motion_data = f['data'][:]
                
                # Downsample and reshape
                motion_data = motion_data[::4]  # 120Hz to 30Hz
                
                if len(motion_data.shape) == 2:
                    frames, total_dims = motion_data.shape
                    joints = total_dims // 3
                    motion_data = motion_data.reshape(frames, joints, 3)
                
                T, joints, coords = motion_data.shape
                motion_flattened = motion_data.reshape(T, joints * coords)
                
                N = motion_flattened.shape[0]
                min_length = self.mogaze_motion_input_length + self.mogaze_motion_target_length
                
                if N >= min_length:
                    self.mogaze_seqs.append(motion_flattened)
                    
                    # For evaluation, use fixed intervals
                    valid_frames = np.arange(0, N - min_length + 1, 10)  # Every 10 frames
                    self.data_idx.extend(zip([idx] * len(valid_frames), valid_frames.tolist()))
                    idx += 1
    
    def __getitem__(self, index):
        idx, start_frame = self.data_idx[index]
        
        end_frame = start_frame + self.mogaze_motion_input_length + self.mogaze_motion_target_length
        motion = self.mogaze_seqs[idx][start_frame:end_frame]
        
        # Don't divide by 1000 - keep same scale as training data
        mogaze_motion_input = motion[:self.mogaze_motion_input_length]
        mogaze_motion_target = motion[self.mogaze_motion_input_length:]
        
        # Ensure we return PyTorch tensors
        return torch.tensor(mogaze_motion_input).float(), torch.tensor(mogaze_motion_target).float()