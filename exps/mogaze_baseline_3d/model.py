import copy

import torch
from torch import nn
from mlp import build_mlps
from einops.layers.torch import Rearrange

class siMLPe(nn.Module):
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        super(siMLPe, self).__init__()
        seq = self.config.motion_mlp.seq_len
        self.arr0 = Rearrange('b n d -> b d n')
        self.arr1 = Rearrange('b d n -> b n d')

        self.motion_mlp = build_mlps(self.config.motion_mlp)

        self.temporal_fc_in = config.motion_fc_in.temporal_fc
        self.temporal_fc_out = config.motion_fc_out.temporal_fc
        
        # FIX: Use MoGaze config keys instead of H36M keys
        if self.temporal_fc_in:
            input_length = self.config.motion.mogaze_input_length_dct
            self.motion_fc_in = nn.Linear(input_length, input_length)
        else:
            self.motion_fc_in = nn.Linear(self.config.motion.dim, self.config.motion.dim)
            
        if self.temporal_fc_out:
            input_length = self.config.motion.mogaze_input_length_dct
            self.motion_fc_out = nn.Linear(input_length, input_length)
        else:
            self.motion_fc_out = nn.Linear(self.config.motion.dim, self.config.motion.dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.motion_fc_out.weight, gain=1e-8)
        nn.init.constant_(self.motion_fc_out.bias, 0)

    def forward(self, motion_input):

        if self.temporal_fc_in:
            motion_feats = self.arr0(motion_input)
            motion_feats = self.motion_fc_in(motion_feats)
        else:
            motion_feats = self.motion_fc_in(motion_input)
            motion_feats = self.arr0(motion_feats)

        motion_feats = self.motion_mlp(motion_feats)

        if self.temporal_fc_out:
            motion_feats = self.motion_fc_out(motion_feats)
            motion_feats = self.arr1(motion_feats)
        else:
            motion_feats = self.arr1(motion_feats)
            motion_feats = self.motion_fc_out(motion_feats)

        return motion_feats