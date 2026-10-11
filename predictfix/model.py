"""SideGRU: per-step side probabilities for one vehicle-obstacle pair."""

import time

import numpy as np
import torch
import torch.nn as nn

from .features import FEATURE_NAMES


class SideGRU(nn.Module):
    """28 -> 80 -> 80 MLP embedding, a two-layer bidirectional GRU with 96
    units per direction, and a linear read-out to four side logits per step."""

    def __init__(self, n_features=len(FEATURE_NAMES)):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(n_features, 80), nn.ReLU(),
            nn.Linear(80, 80), nn.ReLU(),
        )
        self.gru = nn.GRU(80, 96, num_layers=2, dropout=0.10,
                          batch_first=True, bidirectional=True)
        self.out = nn.Linear(192, 4)

    def forward(self, x):
        h, _ = self.gru(self.embed(x))
        return self.out(h)


class Predictor:
    """A trained SideGRU together with its input normalisation."""

    def __init__(self, net, mean, std, device="cpu"):
        self.net = net.to(device).eval()
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.device = device

    @classmethod
    def load(cls, path, device="cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=True)
        net = SideGRU(int(ckpt["n_features"]))
        net.load_state_dict(ckpt["state_dict"])
        return cls(net, ckpt["mean"].numpy(), ckpt["std"].numpy(), device)

    def save(self, path, **metadata):
        torch.save({
            "state_dict": {k: v.detach().cpu() for k, v in self.net.state_dict().items()},
            "mean": torch.from_numpy(self.mean),
            "std": torch.from_numpy(self.std),
            "n_features": len(self.mean),
            "feature_names": FEATURE_NAMES,
            **metadata,
        }, path)

    def __call__(self, sequences):
        """Predicted side, its probability, and the inference time.

        Sequences are run one at a time, as in the timed experiments.
        """
        start = time.perf_counter()
        probs = []
        with torch.no_grad():
            for seq in sequences:
                z = torch.tensor((seq - self.mean) / self.std,
                                 dtype=torch.float32, device=self.device)
                probs.append(torch.softmax(self.net(z), dim=-1).cpu().numpy())
        probs = np.concatenate(probs, axis=0)
        side = probs.argmax(axis=1)
        confidence = probs[np.arange(len(side)), side]
        return side, confidence, time.perf_counter() - start
