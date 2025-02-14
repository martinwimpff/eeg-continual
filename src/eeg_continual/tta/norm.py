from copy import deepcopy

import numpy as np
import torch
from torch import nn

from .alignment import Alignment
from .base import TTAMethod
from eeg_continual.models.custom_bn import MaskedBatchNorm2d


class RobustBN(nn.Module):
    @staticmethod
    def find_bns(parent, alpha):
        replace_mods = []
        if parent is None:
            return []
        for name, child in parent.named_children():
            if isinstance(child, nn.BatchNorm2d) or isinstance(child, MaskedBatchNorm2d):
                module = RobustBN(child, alpha)
                replace_mods.append((parent, name, module))
            else:
                replace_mods.extend(RobustBN.find_bns(child, alpha))

        return replace_mods

    @staticmethod
    def adapt_model(model, alpha):
        replace_mods = RobustBN.find_bns(model, alpha)
        print(f"| Found {len(replace_mods)} modules to be replaced.")
        for (parent, name, child) in replace_mods:
            setattr(parent, name, child)
        return model

    def __init__(self, bn_layer: nn.BatchNorm2d, momentum):
        super(RobustBN, self).__init__()
        self.num_features = bn_layer.num_features
        self.momentum = momentum
        if bn_layer.track_running_stats and bn_layer.running_var is not None and bn_layer.running_mean is not None:
            self.register_buffer("source_mean", deepcopy(bn_layer.running_mean))
            self.register_buffer("source_var", deepcopy(bn_layer.running_var))
            self.source_num = bn_layer.num_batches_tracked

        self.weight = deepcopy(bn_layer.weight)
        self.bias = deepcopy(bn_layer.bias)
        self.eps = bn_layer.eps

    def forward(self, x, trial_lengths=None):
        if self.training:
            b_var, b_mean = torch.var_mean(x, dim=[0, 2, 3], unbiased=False, keepdim=False)  # (C,)
            mean = (1 - self.momentum) * self.source_mean + self.momentum * b_mean
            var = (1 - self.momentum) * self.source_var + self.momentum * b_var
            self.source_mean, self.source_var = deepcopy(mean.detach()), deepcopy(var.detach())
            mean, var = mean.view(1, -1, 1, 1), var.view(1, -1, 1, 1)
        else:
            mean, var = self.source_mean.view(1, -1, 1, 1), self.source_var.view(1, -1, 1, 1)

        x = (x - mean) / torch.sqrt(var + self.eps)
        weight = self.weight.view(1, -1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1)

        return x * weight + bias


class Norm(TTAMethod):
    def __init__(self, model: nn.Module, config: dict, reference: np.ndarray = None):
        super(Norm, self).__init__(model, config)
        if self.config.get("alignment", False):
            self.reference = reference
            self.counter = 0

    @torch.no_grad()
    def forward_and_adapt(self, x):
        if self.config.get("alignment", False):
            x = Alignment.align_data(self, x, self.config.get("alignment"))
        outputs = self.model(x)
        return outputs

    def configure_model(self):
        self.model.eval()
        self.model.requires_grad_(False)

        self.model = RobustBN.adapt_model(
            self.model, alpha=self.config.get("alpha", 0.001))
