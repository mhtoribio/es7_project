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

        frames = 11
        if frames % 2 == 0:
            log.error(f"Only odd frame numbers are supported for kernel size")
        if num_mics % 2 == 0:
            log.error(f"Only odd number of mics are supported for kernel size")
        framepad = (frames - 1) // 2
        micpad = (num_mics - 1) // 2

        self.conv1 = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_hidden,
            kernel_size=(frames, num_mics),   # frames x mic
            padding=(framepad, micpad),
        )
        self.conv2 = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_hidden,
            kernel_size=(frames, num_mics),   # frames x mic
            padding=(framepad, micpad),
        )
        self.conv3 = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_hidden,
            kernel_size=(frames, num_mics),   # frames x mic
            padding=(framepad, micpad),
        )
        self.conv4 = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_hidden,
            kernel_size=(frames, num_mics),   # frames x mic
            padding=(framepad, micpad),
        )
        self.conv5 = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_hidden,
            kernel_size=(frames, num_mics),   # frames x mic
            padding=(framepad, micpad),
        )
        self.conv6 = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_hidden,
            kernel_size=(frames, num_mics),   # frames x mic
            padding=(framepad, micpad),
        )
        self.conv7 = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_hidden,
            kernel_size=(frames, num_mics),   # frames x mic
            padding=(framepad, micpad),
        )
        self.conv8 = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_hidden,
            kernel_size=(frames, num_mics),   # frames x mic
            padding=(framepad, micpad),
        )
        self.conv9 = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_hidden,
            kernel_size=(frames, num_mics),   # frames x mic
            padding=(framepad, micpad),
        )
        self.conv10 = nn.Conv2d(
            in_channels=c_hidden,
            out_channels=c_hidden,
            kernel_size=(frames, num_mics),
            padding=(framepad, 0),
        )
        self.conv11 = nn.Conv1d(
            in_channels=c_hidden,
            out_channels=c_out,
            kernel_size=1,
        )

    def forward(self, x_ctx):
        # x_ctx: (B, 2K, L, M)
        # output: (B, K, L, 1)
        x = F.relu(self.conv1(x_ctx))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x)).squeeze(-1)
        x = self.conv11(x)
        return x
