import torch.nn as nn

class Charbonnier_L1(nn.Module):
    def __init__(self):
        super(Charbonnier_L1, self).__init__()

    def forward(self, diff, mask=None):
        if mask is None:
            loss = ((diff ** 2 + 1e-6) ** 0.5).mean()
        else:
            loss = (((diff ** 2 + 1e-6) ** 0.5) * mask).mean() / (mask.mean() + 1e-9)
        return loss
