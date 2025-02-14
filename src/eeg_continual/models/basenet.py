from einops.layers.torch import Rearrange
import torch
from torch import nn

from .classification_module import ClassificationModule
from .custom_bn import MaskedBatchNorm2d


class _InputBlock(nn.Module):
    def __init__(self,
                 n_channels: int = 24,
                 n_temporal_filters: int = 40,
                 temp_filter_length_inp: int = 25,
                 spatial_expansion: int = 1,
                 pool_length_inp: int = 5,
                 dropout_inp: int = 0.5,
                 padding_mode: str = "zeros",
                 sfreq: int = 250):
        super(_InputBlock, self).__init__()
        self.rearrange_input = Rearrange("b c t -> b 1 c t")
        self.temp_cov = nn.Conv2d(
            1, n_temporal_filters, (1, temp_filter_length_inp),
            padding=(0, temp_filter_length_inp // 2), bias=False,
            padding_mode=padding_mode
        )
        self.bn1 = MaskedBatchNorm2d(n_temporal_filters, sfreq=sfreq)
        self.spat_conv = nn.Conv2d(
            n_temporal_filters, n_temporal_filters * spatial_expansion,
            (n_channels, 1), groups=n_temporal_filters, bias=False)
        self.bn2 = MaskedBatchNorm2d(n_temporal_filters * spatial_expansion,
                                     sfreq=sfreq)
        self.act = nn.ELU()
        self.pool = nn.AvgPool2d((1, pool_length_inp), (1, pool_length_inp))
        self.dropout = nn.Dropout(dropout_inp)

    def forward(self, x, trial_lengths=None):
        x = self.rearrange_input(x)
        x = self.temp_cov(x)
        x = self.bn1(x, trial_lengths)
        x = self.spat_conv(x)
        x = self.bn2(x, trial_lengths)
        x = self.act(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class _ChannelExpansion(nn.Module):
    def __init__(self,
                 n_filters: int = 40,
                 ch_dim: int = 16,
                 sfreq: int = 50):
        super(_ChannelExpansion, self).__init__()
        self.conv = nn.Conv2d(
            n_filters, ch_dim, (1, 1), bias=False)
        self.bn = MaskedBatchNorm2d(ch_dim, sfreq=sfreq)
        self.act = nn.ELU()

    def forward(self, x, trial_lengths=None):
        x = self.conv(x)
        x = self.bn(x, trial_lengths)
        x = self.act(x)
        return x


class _FeatureExtractor(nn.Module):
    def __init__(self,
                 ch_dim: int = 16,
                 temp_filter_length: int = 15,
                 padding_mode: str = "zeros",
                 sfreq: int = 50,
                 pool_length: int = 50,
                 pool_stride: int = 2,
                 dropout: float = 0.5):
        super(_FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(
            ch_dim, ch_dim, (1, temp_filter_length),
            padding=(0, temp_filter_length // 2), bias=False, groups=ch_dim,
            padding_mode=padding_mode)
        self.conv2 = nn.Conv2d(ch_dim, ch_dim, (1, 1), bias=False)
        self.bn = MaskedBatchNorm2d(ch_dim, sfreq=sfreq)
        self.act = nn.ELU()
        self.pool = nn.AvgPool2d((1, pool_length), (1, pool_stride))
        self.dropout = nn.Dropout(dropout)
        self.rearrange_outputs = Rearrange("b f 1 t -> b t f")

    def forward(self, x, trial_lengths=None):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x, trial_lengths)
        x = self.act(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.rearrange_outputs(x)
        return x


class BaseNetModule(nn.Module):
    def __init__(self,
                 n_channels: int = 24,
                 n_temporal_filters: int = 40,
                 temp_filter_length_inp: int = 25,
                 spatial_expansion: int = 1,
                 pool_length_inp: int = 5,
                 dropout_inp: int = 0.5,
                 ch_dim: int = 16,
                 temp_filter_length: int = 15,
                 dropout: float = 0.5,
                 padding_mode: str = "zeros",
                 n_classes: int = 2,
                 sfreq: int = 250,
                 window_length: float = 1.0,
                 f_update: int = 25):
        super(BaseNetModule, self).__init__()
        assert pool_length_inp in [1, 2, 5, 10]
        self.input_block = _InputBlock(
            n_channels=n_channels,
            n_temporal_filters=n_temporal_filters,
            temp_filter_length_inp=temp_filter_length_inp,
            spatial_expansion=spatial_expansion,
            pool_length_inp=pool_length_inp,
            dropout_inp=dropout_inp,
            padding_mode=padding_mode,
            sfreq=sfreq
        )

        f_inter = int(sfreq / pool_length_inp)
        pool_length, pool_stride = int(f_inter * window_length), int(
            f_inter / f_update)

        self.channel_expansion = _ChannelExpansion(
            n_filters=n_temporal_filters * spatial_expansion,
            ch_dim=ch_dim,
            sfreq=f_inter
        )

        self.fe = _FeatureExtractor(
            ch_dim=ch_dim,
            temp_filter_length=temp_filter_length,
            padding_mode=padding_mode,
            sfreq=f_inter,
            pool_length=pool_length,
            pool_stride=pool_stride,
            dropout=dropout
        )

        self.classifier = nn.Linear(ch_dim, n_classes if n_classes > 2 else 1)

    def forward(self, x: torch.tensor, trial_lengths=None):
        x = self.input_block(x, trial_lengths)
        x = self.channel_expansion(x, trial_lengths)
        x = self.fe(x, trial_lengths)
        return self.classifier(x)


class BaseNet(ClassificationModule):
    def __init__(self,
                 n_channels: int = 24,
                 n_temporal_filters: int = 40,
                 temp_filter_length_inp: int = 25,
                 spatial_expansion: int = 1,
                 pool_length_inp: int = 5,
                 dropout_inp: int = 0.5,
                 ch_dim: int = 16,
                 temp_filter_length: int = 15,
                 dropout: float = 0.5,
                 padding_mode: str = "zeros",
                 n_classes: int = 2,
                 sfreq: int = 250,
                 window_length: float = 1.0,
                 f_update: int = 25,
                 **kwargs):
        model = BaseNetModule(
            n_channels=n_channels,
            n_temporal_filters=n_temporal_filters,
            temp_filter_length_inp=temp_filter_length_inp,
            spatial_expansion=spatial_expansion,
            pool_length_inp=pool_length_inp,
            dropout_inp=dropout_inp,
            ch_dim=ch_dim,
            temp_filter_length=temp_filter_length,
            dropout=dropout,
            padding_mode=padding_mode,
            n_classes=n_classes,
            sfreq=sfreq,
            window_length=window_length,
            f_update=f_update
        )
        super(BaseNet, self).__init__(model, n_classes=n_classes, **kwargs)
