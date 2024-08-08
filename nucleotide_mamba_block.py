from mamba_ssm import Mamba
from Mambaout import Mambaout
from conv import ConvModule
import torch.nn as nn
import torch


class swish(nn.Module):
    def __init__(self):
        super(swish, self).__init__()

    def forward(self, x):
        return (x * torch.exp(x)) / (torch.exp(x) + 1)


class conv_mamba_block(nn.Module):
    def __init__(self, d_model, d_hidden, drop_out_rate, feedforward_factor, n_heads, seq_len):
        super(conv_mamba_block, self).__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.drop_out_rate = drop_out_rate
        self.feedforward_factor = feedforward_factor
        self.n_heads = n_heads
        self.seq_len = seq_len
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.mamba = Mamba(
        d_model=self.d_model,
        d_state=16,
        d_conv=4,
        expand=2,
                )
        self.conv = ConvModule(self.d_model)
        self.Mambaout = Mambaout(self.d_model, self.d_hidden, self.d_model, self.drop_out_rate, self.feedforward_factor)
        
    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mamba(x)
        x = self.conv(x)
        x = self.layer_norm(x)
        x = self.Mambaout(x)
        return x
        
