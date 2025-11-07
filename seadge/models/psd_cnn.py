import torch
import torch.nn as nn
import torch.nn.functional as F

from seadge.utils.log import log

class SimplePSDCNN(nn.Module):
    def __init__(self, num_freqbins, num_mics):
        super().__init__()
        self.num_freqbins = num_freqbins
        self.num_mics = num_mics

        c_in  = 2 * num_freqbins
        c_out = num_freqbins
        c_hidden = c_in

        self.conv1 = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_hidden,
            kernel_size=(5, num_mics),   # frames x mic
            padding=(2, 0),
        )
        self.conv2 = nn.Conv1d(
            in_channels=c_hidden,
            out_channels=c_hidden,
            kernel_size=5,
            padding=2,
        )
        self.conv3 = nn.Conv1d(
            in_channels=c_hidden,
            out_channels=c_out,
            kernel_size=1,
        )

    def forward(self, x_ctx):
        # x_ctx: (B, 2K, L, M)
        # output: (B, K, L, 1)
        log.debug(f"input shape: {x_ctx.shape}")
        x = F.relu(self.conv1(x_ctx)).squeeze(3)
        log.debug(f"hidden 1 shape: {x.shape}")
        x = F.relu(self.conv2(x))
        log.debug(f"hidden 2 shape: {x.shape}")
        x = F.softplus(self.conv3(x))
        log.debug(f"output shape: {x.shape}")
        return x

