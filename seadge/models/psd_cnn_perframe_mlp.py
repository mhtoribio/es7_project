import torch
import torch.nn as nn
import torch.nn.functional as F

from seadge.utils.log import log

class SimplePSDCNN(nn.Module):
    def __init__(self, num_freqbins, num_mics):
        super().__init__()
        self.num_freqbins = num_freqbins
        self.num_mics = num_mics

        c_in  = 2 * num_freqbins   # 2K
        c_out = num_freqbins       # K
        c_hidden = c_in

        frames = 21
        if frames % 2 == 0:
            log.error(f"Only odd frame numbers are supported for kernel size")
        if num_mics % 2 == 0:
            log.error(f"Only odd number of mics are supported for kernel size")
        framepad = (frames - 1) // 2
        micpad = (num_mics - 1) // 2

        # convs over context (time × mics)
        self.conv1 = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_hidden,
            kernel_size=(frames, num_mics),
            padding=(framepad, micpad),
        )
        self.conv2 = nn.Conv2d(
            in_channels=c_hidden,
            out_channels=c_hidden,
            kernel_size=(frames, num_mics),
            padding=(framepad, micpad),
        )

        mlp_in_features = c_hidden * num_mics   # (2K * M)
        hidden_mlp = mlp_in_features

        self.per_frame_mlp = nn.Sequential(
            nn.Linear(mlp_in_features, hidden_mlp),
            nn.ReLU(),
            nn.Linear(hidden_mlp, c_out),   # output K per frame
        )

    def forward(self, x_ctx):
        # x_ctx: (B, 2K, L, M)
        # target output: (B, K, L, 1)

        x = F.relu(self.conv1(x_ctx))
        x = F.relu(self.conv2(x))
        # x: (B, c_hidden=2K, L, M)

        B, C, L, M = x.shape

        # (B, 2K, L, M) -> (B, L, 2K, M)
        x = x.permute(0, 2, 1, 3).contiguous()

        # (B, L, 2K, M) -> (B, L, 2K*M)
        x = x.view(B, L, C * M)

        # Apply same MLP to each (B, L, :) slice
        # nn.Linear/Sequential works on last dim: (*, in_features) -> (*, out_features)
        x = self.per_frame_mlp(x)      # (B, L, K)

        # Back to (B, K, L, 1)
        x = x.permute(0, 2, 1)         # (B, K, L)

        x = F.softplus(x)
        return x
