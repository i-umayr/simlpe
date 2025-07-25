import os
import glob
import numpy as np
from tqdm import tqdm
import torch
import torch.utils.data as data


class MoGaze3DDataset(data.Dataset):
    """
    MoGaze dataset loader for 3D position data
    Loads preprocessed 3D joint positions (21 joints × 3 coordinates = 63 dims)
    """
    def __init__(self, config, split_name, data_aug=False):
        super(MoGaze3DDataset, self).__init__()
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
        
        # MoGaze 3D positions: 21 joints with 3 coordinates each = 63 dimensions
        self.motion_dim = config.motion.dim  # 63 for 3D positions (was 66 for Euler)
        
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
        # Based on GazeMotion preprocessing: participants with good gaze data
        participants = ['p1_1', 'p1_2', 'p2_1', 'p4_1', 'p5_1', 'p6_1', 'p6_2', 'p7_1', 'p7_3']
        
        # For leave-one-person-out, we'll do a simple split for now
        # You can modify this to implement proper cross-validation
        if self._split_name == 'train':
            return participants[:-2]  # All but last 2 participants
        else:
            return participants[-2:]  # Last 2 participants for testing
    
    def _get_mogaze_files(self):
        """Get MoGaze 3D position files based on participant split"""
        mogaze_files = []
        
        # Look for preprocessed 3D position files
        all_files = glob.glob(os.path.join(self._mogaze_anno_dir, "*_pose_3d.npy"))
        print(f"🔍 Debug: Found {len(all_files)} total 3D position files")
        if len(all_files) > 0:
            print(f"   Example files: {[os.path.basename(f) for f in all_files[:3]]}")
        
        for participant in self.participant_splits:
            # Match files that start with participant ID (e.g., p1_1_pick_001_pose_3d.npy)
            participant_files = [f for f in all_files if os.path.basename(f).startswith(participant)]
            print(f"   Participant {participant}: {len(participant_files)} files")
            mogaze_files.extend(participant_files)
        
        print(f"✅ Found {len(mogaze_files)} 3D position files for {self._split_name} split")
        print(f"   Participants: {self.participant_splits}")
        return mogaze_files
    
    def _load_mogaze_file(self, file_path):
        """Load a single MoGaze 3D position file"""
        try:
            # Load 3D position data
            motion_data = np.load(file_path)  # [frames, 63] - 21 joints × 3 positions
            
            frames, total_dims = motion_data.shape
            
            if total_dims != 63:
                raise ValueError(f"Expected 63 dimensions (21 joints × 3 positions), got {total_dims}")
            
            print(f"Loaded {file_path}: {motion_data.shape[0]} frames, 21 joints (63 dimensions)")
            return motion_data
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def _collect_all(self):
        """Collect all valid motion sequences"""
        self.mogaze_seqs = []
        self.data_idx = []
        idx = 0
        
        for file_path in tqdm(self._mogaze_files, desc="Loading MoGaze 3D files"):
            motion_data = self._load_mogaze_file(file_path)
            
            if motion_data is None:
                continue
            
            N = motion_data.shape[0]
            
            # Skip sequences that are too short
            min_length = self.mogaze_motion_input_length + self.mogaze_motion_target_length
            if N < min_length:
                print(f"Skipping short sequence: {N} < {min_length}")
                continue
            
            # Data is already in 3D position format [frames, 63] for siMLPe
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
        # Data is already in meters (3D positions), so we keep original scale
        mogaze_motion_input = motion[:self.mogaze_motion_input_length]
        mogaze_motion_target = motion[self.mogaze_motion_input_length:]
        
        return torch.tensor(mogaze_motion_input).float(), torch.tensor(mogaze_motion_target).float()


class MoGaze3DEval(data.Dataset):
    """Evaluation dataset for MoGaze 3D positions - similar to H36MEval"""
    def __init__(self, config, split_name):
        super(MoGaze3DEval, self).__init__()
        self._split_name = split_name
        self._mogaze_anno_dir = config.mogaze_anno_dir
        
        self.mogaze_motion_input_length = config.motion.mogaze_input_length
        
        # FIXED: Use the correct config attribute name
        self.mogaze_motion_target_length = config.motion.mogaze_target_length_eval  # Should be 30 for eval
        
        self.motion_dim = config.motion.dim  # 63 for 3D positions
        self.shift_step = config.shift_step
        
        # For evaluation, we might want to use all participants or specific test set
        self.test_participants = ['p7_1', 'p7_3']  # Example test participants
        self._mogaze_files = self._get_test_files()
        self._collect_all()
        self._file_length = len(self.data_idx)
    
    def __len__(self):
        return self._file_length
    
    def _get_test_files(self):
        """Get test files for evaluation"""
        test_files = []
        all_files = glob.glob(os.path.join(self._mogaze_anno_dir, "*_pose_3d.npy"))
        
        print(f"🔍 Debug eval: Found {len(all_files)} total files")
        
        for participant in self.test_participants:
            # Match files that start with participant ID
            participant_files = [f for f in all_files if os.path.basename(f).startswith(participant)]
            print(f"   Eval participant {participant}: {len(participant_files)} files")
            test_files.extend(participant_files)
        
        print(f"✅ Eval: Found {len(test_files)} files")
        return test_files
    
    def _collect_all(self):
        """Collect evaluation sequences"""
        self.mogaze_seqs = []
        self.data_idx = []
        idx = 0
        
        for file_path in self._mogaze_files:
            # Load 3D position data
            try:
                motion_data = np.load(file_path)  # [frames, 63]
                
                N = motion_data.shape[0]
                min_length = self.mogaze_motion_input_length + self.mogaze_motion_target_length
                
                if N >= min_length:
                    self.mogaze_seqs.append(motion_data)
                    
                    # For evaluation, use fixed intervals
                    valid_frames = np.arange(0, N - min_length + 1, 10)  # Every 10 frames
                    self.data_idx.extend(zip([idx] * len(valid_frames), valid_frames.tolist()))
                    idx += 1
                    
            except Exception as e:
                print(f"Error loading evaluation file {file_path}: {e}")
                continue
    
    def __getitem__(self, index):
        idx, start_frame = self.data_idx[index]
        
        end_frame = start_frame + self.mogaze_motion_input_length + self.mogaze_motion_target_length
        motion = self.mogaze_seqs[idx][start_frame:end_frame]
        
        # Data is already in meters (3D positions), keep same scale as training data
        mogaze_motion_input = motion[:self.mogaze_motion_input_length]
        mogaze_motion_target = motion[self.mogaze_motion_input_length:]
        
        # Ensure we return PyTorch tensors
        return torch.tensor(mogaze_motion_input).float(), torch.tensor(mogaze_motion_target).float()