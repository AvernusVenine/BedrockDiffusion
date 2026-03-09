import torch

class CosineScheduler:

    def __init__(self, steps):
        self.steps = steps
        self.offset = 8e-3
        self.exponent = 2.0

    def compute(self):
        t = torch.arange(self.steps + 2)
        delta = ((t / (self.steps + 2) + self.offset) / (1 + self.offset) * torch.pi / 2).cos() ** self.exponent

        alpha = torch.clip(delta[1:] / delta[:-1], 1e-3, 1)
        return alpha