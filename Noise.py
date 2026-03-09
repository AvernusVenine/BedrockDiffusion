import torch
import numpy as np

class GaussianNoise:

    def __init__(self, data):
        self.noise = None
        self.create_noise(data)

    def create_noise(self, data):
        self.noise = torch.randn_like(data)

    def apply(self, alpha, prev_alpha, data):
        noisy = np.sqrt(alpha) * data + np.sqrt(1 - alpha) * self.noise
        prev_noisy = np.sqrt(prev_alpha) * data + np.sqrt(1 - alpha) * self.noise

        return noisy, noisy - prev_noisy