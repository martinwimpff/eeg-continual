import torch
from torch import nn


class TTAMethod(nn.Module):
    def __init__(self, model: nn.Module, config: dict):
        super(TTAMethod, self).__init__()
        self.model = model
        self.config = config
        self.device = self.model.device

        self.configure_model()

    def forward(self, x):
        assert x.shape[0] == 1  # only single-sample TTA allowed
        return self.forward_and_adapt(x)

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        raise NotImplementedError

    def configure_model(self):
        raise NotImplementedError
