import torch
import torch.nn as nn
import torch.nn.functional as F

from seadge.utils.log import log

class PSD_CNN_GRU(nn.Module):
    def __init__(
        self,
        num_freqbins: int,
        num_mics: int,
        *,
        frames: int = 11,             # temporal kernel (must be odd)
        gru_hidden: int | None = None, # None -> use c_hidden
        gru_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_freqbins = num_freqbins
        self.num_mics = num_mics

        c_in  = 2 * num_freqbins         # real/imag per freq bin as channels
        c_out = num_freqbins             # one PSD channel per freq bin
        c_hidden = c_in                  # keep hidden same as input channels

        if frames % 2 == 0:
            log.error("Only odd frame numbers are supported for kernel size")
        if num_mics % 2 == 0:
            log.error("Only odd number of mics are supported for kernel size")
        framepad = (frames - 1) // 2
        micpad   = (num_mics - 1) // 2

        # (B, 2K, L, M) --conv2d(frames×M)--> (B, c_hidden, L, M)
        self.conv1 = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_hidden,
            kernel_size=(frames, num_mics),   # frames × mic
            padding=(framepad, micpad),
        )
        # (B, c_hidden, L, M) --conv2d(frames×M, no mic pad)--> (B, c_hidden, L, 1)
        self.conv2 = nn.Conv2d(
            in_channels=c_hidden,
            out_channels=c_hidden,
            kernel_size=(frames, num_mics),
            padding=(framepad, 0),
        )

        # GRU over time L; input is (B, L, c_hidden) with batch_first=True
        gru_in = c_hidden
        gru_h  = gru_hidden or c_hidden
        self.gru = nn.GRU(
            input_size=gru_in,
            hidden_size=gru_h,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)

        gru_out_ch = gru_h * (2 if bidirectional else 1)

        # 1×1 temporal conv to map GRU features -> K PSD channels, keeps length L
        self.conv3 = nn.Conv1d(
            in_channels=gru_out_ch,
            out_channels=c_out,
            kernel_size=1,
        )

    def forward(self, x_ctx: torch.Tensor) -> torch.Tensor:
        """
        x_ctx: (B, 2K, L, M)  [channels = 2×freqbins]
        returns: (B, K, L)    [PSD per freq bin over time]
        """
        x = F.relu(self.conv1(x_ctx))              # (B, c_hidden, L, M)
        x = F.relu(self.conv2(x)).squeeze(-1)      # (B, c_hidden, L)

        # GRU expects (B, L, C); then back to (B, C, L) for conv1d
        x = x.transpose(1, 2)                      # (B, L, c_hidden)
        x, _ = self.gru(x)                         # (B, L, gru_out_ch)
        x = self.dropout(x)
        x = x.transpose(1, 2)                      # (B, gru_out_ch, L)

        x = self.conv3(x)                          # (B, K, L)

        # If you want strictly non-negative PSDs, uncomment:
        # x = F.softplus(x)
        return x
