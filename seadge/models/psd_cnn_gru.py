import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------
# Shared per-mic temporal GRU-CNN branch
# --------------------------------------
class GRUCNNBranchTime(nn.Module):
    """
    Input per mic : (BM, 2K, L)
    Output per mic: (BM, K,  L)
    - Conv1d over time for local temporal smoothing
    - GRU over time for sequence modeling
    - Linear map to K PSD channels per frame
    """
    def __init__(
        self,
        K: int,
        *,
        frames: int = 5,         # odd kernel size (temporal)
        c_branch: int | None = None,
        gru_hidden: int | None = None,
        gru_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        if frames % 2 == 0:
            raise ValueError("frames must be odd")
        self.K = K

        C_in = 2 * K
        Cb   = c_branch or C_in
        gh   = gru_hidden or Cb
        padt = (frames - 1) // 2

        # temporal conv: (BM, 2K, L) -> (BM, Cb, L)
        self.conv_t = nn.Conv1d(
            in_channels=C_in, out_channels=Cb,
            kernel_size=frames, padding=padt
        )

        # GRU over time: (BM, L, Cb) -> (BM, L, gh*)
        self.gru = nn.GRU(
            input_size=Cb, hidden_size=gh,
            num_layers=gru_layers, batch_first=True,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)

        # project back to K per frame
        self.proj = nn.Linear(gh * (2 if bidirectional else 1), K)

    def forward(self, x_bm: torch.Tensor) -> torch.Tensor:
        # x_bm: (BM, 2K, L)
        z = F.relu(self.conv_t(x_bm))         # (BM, Cb, L)
        z = z.transpose(1, 2)                 # (BM, L, Cb)
        g, _ = self.gru(z)                    # (BM, L, gh*)
        g = self.dropout(g)
        p = self.proj(g)                      # (BM, L, K)
        y = p.transpose(1, 2).contiguous()    # (BM, K, L)
        return y


# ----------------------
# Mic attention fusion
# ----------------------
class MicAttentionFusion(nn.Module):
    """
    Fuse per-mic features via attention over M.
    Input : (B, M, K, L)
    Output: (B, K, L)
    """
    def __init__(self):
        super().__init__()
        # score net over (K,L) → scalar per mic (implemented as 1x1 conv on a fake channel)
        # we use a tiny MLP via convs for stability
        self.scorer = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(4, 1, kernel_size=1),
        )

    def forward(self, z_bm: torch.Tensor) -> torch.Tensor:
        # z_bm: (B, M, K, L)
        B, M, K, L = z_bm.shape
        z = z_bm.view(B * M, 1, K, L)                # (B*M, 1, K, L)
        s = self.scorer(z).view(B, M, 1, K, L)       # (B, M, 1, K, L)
        w = torch.softmax(s, dim=1)                  # across mics
        fused = (w * z_bm.view(B, M, K, L).unsqueeze(2)).sum(dim=1)  # (B, K, L)
        return fused


# ------------------------------------------
# Parallel per-mic GRU-CNN + attention → PSD
# ------------------------------------------
class PSD_CNN_GRU(nn.Module):
    """
    Input : (B, 2K, L, M)   [RI split across freq into channels]
    Output: (B, K, L)       [PSD per freq bin over time]
    """
    def __init__(
        self,
        num_freqbins: int, # K
        num_mics: int, # M
        *,
        frames: int = 5,
        c_branch: int | None = None,
        gru_hidden: int | None = None,
        gru_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
        nonneg: bool = True,   # Softplus at the end
    ):
        super().__init__()
        self.K = num_freqbins
        self.M = num_mics
        self.nonneg = nonneg

        self.branch = GRUCNNBranchTime(
            num_freqbins,
            frames=frames,
            c_branch=c_branch,
            gru_hidden=gru_hidden,
            gru_layers=gru_layers,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        self.fuse = MicAttentionFusion()

    def forward(self, x_ctx: torch.Tensor) -> torch.Tensor:
        """
        x_ctx: (B, 2K, L, M)
        returns: (B, K, L)
        """
        B, C, L, M = x_ctx.shape
        assert M == self.M and C == 2 * self.K, "bad input shape"

        # apply shared branch per mic (merge mic into batch)
        x_bm = x_ctx.permute(0, 3, 1, 2).contiguous().view(B * M, C, L)  # (BM, 2K, L)
        z_bm = self.branch(x_bm).view(B, M, self.K, L)                   # (B, M, K, L)

        phi = self.fuse(z_bm)                                            # (B, K, L)
        if self.nonneg:
            phi = F.softplus(phi)
        return phi
