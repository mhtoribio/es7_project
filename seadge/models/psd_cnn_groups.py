import torch
import torch.nn as nn
import torch.nn.functional as F

from seadge.utils.log import log

class SimplePSDCNN(nn.Module):
    def __init__(self, num_freqbins: int, num_mics: int, *, frames: int = 11, per_bin_hidden: int = 2):
        super().__init__()
        self.num_freqbins = num_freqbins
        self.num_mics     = num_mics
        K = num_freqbins
        M = num_mics

        if frames % 2 == 0:
            log.error("Only odd frame numbers are supported for kernel size")
        if M % 2 == 0:
            log.error("Only odd number of mics are supported for kernel size")
        framepad = (frames - 1) // 2
        micpad   = (M - 1) // 2

        # Channels: pack bins as groups
        in_ch_total   = 2 * K            # 2 (RI) per bin
        hid_ch_total  = per_bin_hidden * K
        out_ch_total  = K                # 1 per bin

        # (B, 2K, L, M)  -> (B, H*K, L, M)
        self.conv1 = nn.Conv2d(
            in_channels=in_ch_total,
            out_channels=hid_ch_total,
            kernel_size=(frames, M),
            padding=(framepad, micpad),
            groups=K,                    # <<< per-bin independence
        )
        # (B, H*K, L, M) -> (B, H*K, L, 1)
        self.conv2 = nn.Conv2d(
            in_channels=hid_ch_total,
            out_channels=hid_ch_total,
            kernel_size=(frames, M),
            padding=(framepad, 0),
            groups=K,                    # <<< keep bins independent
        )
        # After squeeze: (B, H*K, L) -> (B, K, L)
        self.conv3 = nn.Conv1d(
            in_channels=hid_ch_total,
            out_channels=out_ch_total,
            kernel_size=1,
            groups=K,                    # <<< per bin: H -> 1
        )

    def forward(self, x_ctx: torch.Tensor) -> torch.Tensor:
        # x_ctx: (B, 2K, L, M)  ; returns (B, K, L)
        x = F.relu(self.conv1(x_ctx))
        x = F.relu(self.conv2(x)).squeeze(-1)
        x = self.conv3(x)
        return x
