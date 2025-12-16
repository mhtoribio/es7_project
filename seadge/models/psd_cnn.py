import torch
import torch.nn as nn
import torch.nn.functional as F

from seadge.utils.log import log

class SimplePSDCNN(nn.Module):
    def __init__(self, num_freqbins, num_mics):
        super().__init__()
        self.num_freqbins = num_freqbins

        c_in  = num_freqbins
        c_out = num_freqbins
        c_hidden = c_in

        self.conv1 = nn.Conv2d(
            in_channels=c_hidden,
            out_channels=c_hidden,
            kernel_size=(5, 3),   # frames x mic
            padding=(2, 0),
        )
        self.conv2 = nn.Conv2d(
            in_channels=c_hidden,
            out_channels=c_hidden,
            kernel_size=(11, 3),
            padding=(5, 0),
        )
        self.conv3 = nn.Conv1d(
            in_channels=c_hidden,
            out_channels=c_out,
            kernel_size=1,
        )

    def forward(self, x_ctx):
        # x_ctx: (B, 2K, L, M)
        # output: (B, K, L, 1)
        x = F.relu(self.conv1(x_ctx))
        x = F.relu(self.conv2(x)).squeeze(-1)
        x = self.conv3(x)
        return x
