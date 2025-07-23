import os
import glob
import numpy as np
import torch
import torch.utils.data as data

class MoGazeDataset(data.Dataset):
    """
    MoGaze dataset loader compatible with siMLPe training
    Similar to H36MDataset but for MoGaze motion data
    """
    def __init__(self, config, split_name, data_aug=False):
        super(MoGazeDataset, self).__init__()
        self._split_name = split_name
        self.data_aug = data_aug
        
        # Update paths for MoGaze
        self._mogaze_anno_dir = config.mogaze_anno_dir
        print(f"🔍 Dataset looking for data in: {self._mogaze_anno_dir}")
        
        # Verify directory exists
        if not os.path.exists(self._mogaze_anno_dir):
            raise FileNotFoundError(f"MoGaze directory not found: {self._mogaze_anno_dir}")
        
        self._mogaze_files = self._get_mogaze_files()
        
        self.mogaze_motion_input_length = config.motion.mogaze_input_length  
        self.mogaze_motion_target_length = config.motion.mogaze_target_length
        self.motion_dim = config.motion.dim  # Should be 66 for MoGaze
        self.shift_step = config.shift_step
        
        self._collect_all()
        self._file_length = len(self.data_idx)

    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self._mogaze_files)

    def _get_mogaze_files(self):
        seq_names = []
        
        # Construct file paths
        if self._split_name == 'train':
            train_file = os.path.join(self._mogaze_anno_dir, "mogaze_train.txt")
            print(f"🔍 Looking for train file: {train_file}")
            
            if not os.path.exists(train_file):
                raise FileNotFoundError(f"Train file not found: {train_file}")
                
            with open(train_file, 'r') as f:
                seq_names = [line.strip() for line in f.readlines() if line.strip()]
                
        else:
            test_file = os.path.join(self._mogaze_anno_dir, "mogaze_test.txt")
            print(f"🔍 Looking for test file: {test_file}")
            
            if not os.path.exists(test_file):
                raise FileNotFoundError(f"Test file not found: {test_file}")
                
            with open(test_file, 'r') as f:
                seq_names = [line.strip() for line in f.readlines() if line.strip()]

        print(f"📄 Found {len(seq_names)} file entries for {self._split_name}")

        file_list = []
        for dataset in seq_names:
            file_path = os.path.join(self._mogaze_anno_dir, dataset)
            if os.path.exists(file_path):
                file_list.append(file_path)
            else:
                print(f"⚠️  File not found: {file_path}")

        print(f"📁 Valid files found: {len(file_list)}")

        mogaze_files = []
        for path in file_list:
            # Memory-safe loading with subsampling
            try:
                motion_data = []
                frames_processed = 0
                frames_kept = 0
                
                print(f"📄 Processing: {os.path.basename(path)}")
                
                with open(path, 'r') as f:
                    for line_idx, line in enumerate(f):
                        frames_processed += 1
                        
                        # Sample every 10th frame to reduce memory usage
                        if line_idx % 10 == 0:
                            line = line.strip()
                            if line:
                                values = [float(x) for x in line.split(',')]
                                if len(values) == 66:  # Ensure correct dimension
                                    motion_data.append(values)
                                    frames_kept += 1
                        
                        # Safety limit: max 100k frames per file to prevent memory overflow
                        if frames_kept >= 100000:
                            print(f"⚠️  Reached safety limit (100k frames) for {os.path.basename(path)}")
                            break
                        
                        # Progress indicator for very large files
                        if frames_processed % 500000 == 0:
                            print(f"    Processed {frames_processed:,} frames, kept {frames_kept:,}")
                
                if motion_data:
                    motion_array = np.array(motion_data)
                    mogaze_files.append(motion_array)
                    
                    reduction_factor = frames_processed / frames_kept if frames_kept > 0 else 1
                    print(f"✅ Loaded {frames_kept:,} frames from {os.path.basename(path)} "
                        f"(reduced from {frames_processed:,}, {reduction_factor:.1f}x smaller)")
                else:
                    print(f"⚠️  No valid data found in {os.path.basename(path)}")
                    
            except Exception as e:
                print(f"❌ Error loading {path}: {e}")
                continue

        total_frames = sum(len(arr) for arr in mogaze_files)
        print(f"✅ Successfully loaded {len(mogaze_files)} MoGaze files for {self._split_name}")
        print(f"📊 Total frames across all files: {total_frames:,}")
        
        # Memory usage estimation
        memory_gb = total_frames * 66 * 8 / (1024**3)  # 8 bytes per float64
        print(f"💾 Estimated memory usage: {memory_gb:.2f} GB")
        
        return mogaze_files

    def _collect_all(self):
        """Collect all sequences similar to H36M dataset"""
        self.mogaze_seqs = []
        self.data_idx = []
        idx = 0
        
        total_sequences = 0
        
        for motion_poses in self._mogaze_files:
            N = len(motion_poses)
            if N < self.mogaze_motion_target_length + self.mogaze_motion_input_length:
                print(f"⚠️  Skipping short sequence: {N} frames < {self.mogaze_motion_target_length + self.mogaze_motion_input_length} required")
                continue
                
            # No additional sampling since we already processed in converter
            T = motion_poses.shape[0]
            motion_poses = motion_poses.reshape(T, -1)
            
            self.mogaze_seqs.append(motion_poses)
            valid_frames = np.arange(0, T - self.mogaze_motion_input_length - self.mogaze_motion_target_length + 1, self.shift_step)
            
            self.data_idx.extend(zip([idx] * len(valid_frames), valid_frames.tolist()))
            total_sequences += len(valid_frames)
            idx += 1
            
        print(f"📊 Total sequences created: {total_sequences}")

    def __getitem__(self, index):
        idx, start_frame = self.data_idx[index]
        frame_indexes = np.arange(start_frame, start_frame + self.mogaze_motion_input_length + self.mogaze_motion_target_length)
        motion = self.mogaze_seqs[idx][frame_indexes]
        
        if self.data_aug:
            if torch.rand(1)[0] > .5:
                idx_reverse = [i for i in range(motion.shape[0]-1, -1, -1)]
                motion = motion[idx_reverse]

        mogaze_motion_input = motion[:self.mogaze_motion_input_length] / 1000  # Convert to meters like H36M
        mogaze_motion_target = motion[self.mogaze_motion_input_length:] / 1000

        mogaze_motion_input = torch.tensor(mogaze_motion_input).float()
        mogaze_motion_target = torch.tensor(mogaze_motion_target).float()
        
        return mogaze_motion_input, mogaze_motion_target