from copy import deepcopy

import torch
from torch import nn


class MaskedBatchNorm2d(nn.BatchNorm2d):
    """
    custom BatchNorm2d to allow training with different trial lengths
    """
    def __init__(
            self,
            num_features: int,
            eps: float = 1e-5,
            momentum: float = 0.1,
            affine: bool = True,
            track_running_stats: bool = True,
            device=None,
            dtype=None,
            sfreq: int = 250
    ):
        super(MaskedBatchNorm2d, self).__init__(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            device=device,
            dtype=dtype)
        self.sfreq = sfreq

    def forward(self, x, trial_lengths=None):
        if self.training:
            if trial_lengths is None:
                # individual windows -> no masking
                b_var, b_mean = torch.var_mean(x, dim=[0, 2, 3], unbiased=False, keepdim=False)  # (C,)
            else:
                # build masks from trial lengths
                trial_length_samples = (self.sfreq * trial_lengths).int()
                trial_length_samples_reshaped = trial_length_samples.view(
                    len(trial_length_samples), 1, 1, 1)
                mask = (torch.arange(x.shape[-1], device=x.device).view(1, 1, 1, -1) < trial_length_samples_reshaped).int()  # Bx1x1xT

                # calculate mean and variance manually
                b_mean = (x * mask).sum(dim=[0, 2, 3]) / (mask.sum() * x.shape[-2])
                b_var = (((x - b_mean.view(1, -1, 1, 1)) ** 2) * mask).sum(
                    dim=[0, 2, 3]) / (mask.sum() * x.shape[-2])

            mean = (1 - self.momentum) * self.running_mean + self.momentum * b_mean
            var = (1 - self.momentum) * self.running_var + self.momentum * b_var
            self.running_mean, self.running_var = deepcopy(mean.detach()), deepcopy(
                var.detach())
            mean, var = mean.view(1, -1, 1, 1), var.view(1, -1, 1, 1)

        else:
            mean, var = self.running_mean.view(1, -1, 1, 1), self.running_var.view(1, -1, 1, 1)

        x = (x - mean) / torch.sqrt(var + self.eps)
        weight = self.weight.view(1, -1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1)

        return x * weight + bias
