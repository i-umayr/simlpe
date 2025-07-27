import os
import glob
import numpy as np
from tqdm import tqdm
import torch
import torch.utils.data as data


class MoGazeGazeDataset(data.Dataset):
    """
    MoGaze dataset loader for 3D position + gaze integration
    Loads 3D joint positions (21 joints × 3 = 63D) + gaze vectors (3D) = 66D total
    """
    def __init__(self, config, split_name, data_aug=False):
        super(MoGazeGazeDataset, self).__init__()
        self._split_name = split_name
        self.data_aug = data_aug
        
        self._mogaze_anno_dir = config.mogaze_anno_dir
        
        # Use the same config names as before for model compatibility
        self.mogaze_motion_input_length = config.motion.mogaze_input_length  # 50 frames
        
        if split_name == 'train':
            self.mogaze_motion_target_length = config.motion.mogaze_target_length_train  # 10 frames
        else:
            self.mogaze_motion_target_length = config.motion.mogaze_target_length_eval   # 30 frames
        
        # NOW: 63 (pose) + 3 (gaze) = 66 dimensions total
        self.motion_dim = config.motion.dim  # Should be 66 for pose+gaze
        
        self.shift_step = config.shift_step
        
        # Get participant splits (same as before - NO data leakage)
        self.participant_splits = self._get_participant_splits()
        self._mogaze_files = self._get_mogaze_files()
        
        self._collect_all()
        self._file_length = len(self.data_idx)
    
    def __len__(self):
        return self._file_length
    
    def _get_participant_splits(self):
        """Define participant splits - SAME AS BEFORE (no data leakage)"""
        participants = ['p1_1', 'p1_2', 'p2_1', 'p4_1', 'p5_1', 'p6_1', 'p6_2', 'p7_1', 'p7_3']
        
        if self._split_name == 'train':
            return participants[:-2]  # ['p1_1', ..., 'p6_2'] - same as before
        else:
            return participants[-2:]  # ['p7_1', 'p7_3'] - same as before
    
    def _get_mogaze_files(self):
        """Get MoGaze pose and gaze file pairs"""
        mogaze_file_pairs = []
        
        # Look for pose files first
        all_pose_files = glob.glob(os.path.join(self._mogaze_anno_dir, "*_pose_3d.npy"))
        print(f"🔍 Found {len(all_pose_files)} total pose files")
        
        for participant in self.participant_splits:
            # Match pose files for this participant
            participant_pose_files = [f for f in all_pose_files 
                                    if os.path.basename(f).startswith(participant)]
            
            # For each pose file, check if corresponding gaze file exists
            valid_pairs = 0
            for pose_file in participant_pose_files:
                gaze_file = pose_file.replace('_pose_3d.npy', '_gaze.npy')
                if os.path.exists(gaze_file):
                    mogaze_file_pairs.append((pose_file, gaze_file))
                    valid_pairs += 1
            
            print(f"   Participant {participant}: {valid_pairs} valid pose-gaze pairs")
        
        print(f"✅ Found {len(mogaze_file_pairs)} total pose-gaze pairs for {self._split_name} split")
        print(f"   Participants: {self.participant_splits}")
        return mogaze_file_pairs
    
    def _load_mogaze_file_pair(self, pose_file, gaze_file):
        """Load a pose-gaze file pair and concatenate them"""
        try:
            # Load 3D position data
            pose_data = np.load(pose_file)  # [frames, 63] - 21 joints × 3 positions
            
            # Load gaze data  
            gaze_data = np.load(gaze_file)  # [frames, 3] - 3D gaze direction vectors
            
            frames_pose, pose_dims = pose_data.shape
            frames_gaze, gaze_dims = gaze_data.shape
            
            # Verify dimensions
            if pose_dims != 63:
                raise ValueError(f"Expected 63 pose dimensions, got {pose_dims}")
            if gaze_dims != 3:
                raise ValueError(f"Expected 3 gaze dimensions, got {gaze_dims}")
            if frames_pose != frames_gaze:
                raise ValueError(f"Frame mismatch: pose {frames_pose} vs gaze {frames_gaze}")
            
            # Concatenate pose + gaze = 66D
            combined_data = np.concatenate([pose_data, gaze_data], axis=1)  # [frames, 66]
            
            return combined_data
                
        except Exception as e:
            print(f"Error loading {pose_file} + {gaze_file}: {e}")
            return None
    
    def _collect_all(self):
        """Collect all valid motion sequences"""
        self.mogaze_seqs = []
        self.data_idx = []
        idx = 0
        
        for pose_file, gaze_file in tqdm(self._mogaze_files, desc="Loading MoGaze pose+gaze files"):
            combined_data = self._load_mogaze_file_pair(pose_file, gaze_file)
            
            if combined_data is None:
                continue
            
            N = combined_data.shape[0]
            
            # Skip sequences that are too short
            min_length = self.mogaze_motion_input_length + self.mogaze_motion_target_length
            if N < min_length:
                continue
            
            # Data is now pose+gaze format [frames, 66] ready for siMLPe
            self.mogaze_seqs.append(combined_data)
            
            # Create valid frame indices
            valid_frames = np.arange(0, N - min_length + 1, self.shift_step)
            self.data_idx.extend(zip([idx] * len(valid_frames), valid_frames.tolist()))
            idx += 1
        
        print(f"Collected {len(self.mogaze_seqs)} sequences with {len(self.data_idx)} samples")
        print(f"Data format: {self.mogaze_seqs[0].shape if self.mogaze_seqs else 'No data'} (66D: 63 pose + 3 gaze)")
    
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


class MoGazeGazeEval(data.Dataset):
    """Evaluation dataset for MoGaze 3D positions + gaze"""
    def __init__(self, config, split_name):
        super(MoGazeGazeEval, self).__init__()
        self._split_name = split_name
        self._mogaze_anno_dir = config.mogaze_anno_dir
        
        self.mogaze_motion_input_length = config.motion.mogaze_input_length
        self.mogaze_motion_target_length = config.motion.mogaze_target_length_eval  # Should be 30 for eval
        
        self.motion_dim = config.motion.dim  # 66 for pose+gaze
        self.shift_step = config.shift_step
        
        # SAME AS TRAINING: Use only p7_1, p7_3 for testing (NO data leakage)
        self.test_participants = ['p7_1', 'p7_3']
        self._mogaze_files = self._get_test_files()
        self._collect_all()
        self._file_length = len(self.data_idx)
    
    def __len__(self):
        return self._file_length
    
    def _get_test_files(self):
        """Get test file pairs for evaluation"""
        test_file_pairs = []
        all_pose_files = glob.glob(os.path.join(self._mogaze_anno_dir, "*_pose_3d.npy"))
        
        print(f"🔍 Debug eval: Found {len(all_pose_files)} total files")
        
        for participant in self.test_participants:
            # Match pose files for this participant
            participant_pose_files = [f for f in all_pose_files 
                                    if os.path.basename(f).startswith(participant)]
            
            # Check for corresponding gaze files
            valid_pairs = 0
            for pose_file in participant_pose_files:
                gaze_file = pose_file.replace('_pose_3d.npy', '_gaze.npy')
                if os.path.exists(gaze_file):
                    test_file_pairs.append((pose_file, gaze_file))
                    valid_pairs += 1
            
            print(f"   Eval participant {participant}: {valid_pairs} files")
        
        print(f"✅ Eval: Found {len(test_file_pairs)} pose-gaze pairs")
        return test_file_pairs
    
    def _collect_all(self):
        """Collect evaluation sequences"""
        self.mogaze_seqs = []
        self.data_idx = []
        idx = 0
        
        for pose_file, gaze_file in self._mogaze_files:
            try:
                # Load and combine pose + gaze
                pose_data = np.load(pose_file)  # [frames, 63]
                gaze_data = np.load(gaze_file)  # [frames, 3]
                
                if pose_data.shape[0] != gaze_data.shape[0]:
                    continue
                
                combined_data = np.concatenate([pose_data, gaze_data], axis=1)  # [frames, 66]
                
                N = combined_data.shape[0]
                min_length = self.mogaze_motion_input_length + self.mogaze_motion_target_length
                
                if N >= min_length:
                    self.mogaze_seqs.append(combined_data)
                    
                    # For evaluation, use fixed intervals
                    valid_frames = np.arange(0, N - min_length + 1, 10)  # Every 10 frames
                    self.data_idx.extend(zip([idx] * len(valid_frames), valid_frames.tolist()))
                    idx += 1
                    
            except Exception as e:
                print(f"Error loading evaluation file pair: {e}")
                continue
    
    def __getitem__(self, index):
        idx, start_frame = self.data_idx[index]
        
        end_frame = start_frame + self.mogaze_motion_input_length + self.mogaze_motion_target_length
        motion = self.mogaze_seqs[idx][start_frame:end_frame]
        
        # Data is already in correct format (66D: 63 pose + 3 gaze)
        mogaze_motion_input = motion[:self.mogaze_motion_input_length]
        mogaze_motion_target = motion[self.mogaze_motion_input_length:]
        
        # Ensure we return PyTorch tensors
        return torch.tensor(mogaze_motion_input).float(), torch.tensor(mogaze_motion_target).float()