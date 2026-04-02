import torch
import torch.nn as nn
import torch.nn.functional as F

class AvgPoolFilter2D(nn.Module):

    def __init__(self, kernel_size, stride, padding, filter_value=-1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.filter_value = filter_value

    def forward(self, X):
        mask = (X != self.filter_value).float()
        X_shifted = (X - self.filter_value) * mask

        sum_values = F.avg_pool2d(X_shifted * mask, self.kernel_size, self.stride, self.padding) * (self.kernel_size ** 2)
        count = F.avg_pool2d(mask, self.kernel_size, self.stride, self.padding) * (self.kernel_size ** 2)

        avg = (sum_values / torch.clamp(count, min=1)) + self.filter_value

        return torch.where(count > 0, avg, X.new_full(avg.shape, self.filter_value))

class BoreholeEncoder(nn.Module):
    """
    Takes in a 256x256 sparse array of borehole readings, downsamples them to make the data dense, then encodes into a
    single vector of length cross_attention_dim for use in cross attention

    (256 x 256 x   C) Sparse readings
    (32  x  32 x   C) Average pool downsample
    (16  x  16 x  64) Convolution
    (8   x   8 x 128) Convolution
    (4   x   4 x 256) Convolution
    (2   x   2 x 512) Convolution
    (1   x   1 x CAD) Convolution

    Flatten into single vector of length cross_attention_dim and return
    """
    def __init__(self, in_channels, cross_attention_dim):
        super().__init__()

        self.avgpool = AvgPoolFilter2D(8, 8, 0, -1)

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(),
            nn.Conv2d(256, cross_attention_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, cross_attention_dim),
            nn.ReLU()
        )

    def forward(self, X):
        X = self.avgpool(X)
        X = self.encoder(X)

        return X


class GeophysicalEncoder(nn.Module):
    """
    Takes in a dense 256x256xC array of geophysical context and encodes it so it can be used as a base for diffusion

    (256 x 256 x    C) Dense context
    (128 x 128 x   64) Convolution
    (64  x  64 x  128) Convolution
    (32  x  32 x  Emb) Convolution
    """

    def __init__(self, in_channels, embedding_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, embedding_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, embedding_dim),
            nn.SiLU(),
        )

    def forward(self, X):
        X = self.encoder(X)
        return X

class GeophysicalDecoder(nn.Module):
    """
    Takes in an embedded patch of geophysical context and decodes it into a (256 x 256 x C) image

    (32  x  32 x Emb) Embedded Context
    (64  x  64 x 128) Transpose Convolution
    (128 x 128 x  64) Transpose Convolution
    (256 x 256 x   C) Transpose Convolution
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.ConvTranspose2d(64, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, X):
        X = self.decoder(X)
        return X

### --- DEPRECIATED --- ###
class Encoder(nn.Module):

    def __init__(self, in_channels, cross_attention_dim, seq_len):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(),
            nn.Conv2d(256, cross_attention_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, cross_attention_dim),
            nn.ReLU()
        )

        size = int(seq_len ** 0.5)
        self.pool = nn.AdaptiveAvgPool2d((size, size))

    def forward(self, X):
        X = self.encoder(X)
        X = self.pool(X)

        B, D, H, W = X.shape
        X = X.view(B, D, H * W)
        X = X.permute(0, 2, 1)

        return X

