from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision import models


@dataclass
class ModelConfig:
    architecture: str = "cnn_lstm"
    num_classes: int = 101
    hidden_size: int = 256


class CNNEncoder(nn.Module):
    def __init__(self, out_dim: int = 512) -> None:
        super().__init__()
        backbone = models.resnet18(weights=None)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.proj = nn.Linear(512, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        x = self.features(x).flatten(1)
        x = self.proj(x)
        return x.view(b, t, -1)


class ViTEncoder(nn.Module):
    def __init__(self, out_dim: int = 512) -> None:
        super().__init__()
        vit = models.vit_b_16(weights=None)
        self.encoder = vit
        self.proj = nn.Linear(1000, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        x = self.encoder(x)
        x = self.proj(x)
        return x.view(b, t, -1)


class TemporalHead(nn.Module):
    def __init__(self, in_dim: int, hidden_size: int, num_classes: int) -> None:
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.rnn(x)
        logits = self.classifier(h_n[-1])
        return logits


class VideoUnderstandingModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.architecture == "vit":
            self.frame_encoder = ViTEncoder(out_dim=config.hidden_size)
        else:
            self.frame_encoder = CNNEncoder(out_dim=config.hidden_size)

        self.temporal_head = TemporalHead(
            in_dim=config.hidden_size,
            hidden_size=config.hidden_size,
            num_classes=config.num_classes,
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        frame_tokens = self.frame_encoder(frames)
        return self.temporal_head(frame_tokens)
