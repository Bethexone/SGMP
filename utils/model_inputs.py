from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ModelInputs:
    frame: torch.Tensor
    frame_sequence: torch.Tensor

    def __post_init__(self) -> None:
        if self.frame.dim() != 4:
            raise ValueError(f"frame must be [B,C,H,W], got {self.frame.shape}")
        if self.frame_sequence.dim() != 5:
            raise ValueError(f"frame_sequence must be [B,C,T,H,W], got {self.frame_sequence.shape}")

    @property
    def num_frames(self) -> int:
        return int(self.frame_sequence.shape[2])

    def decompose(self):
        return self.frame, self.frame_sequence

    def to(self, device):
        self.frame = self.frame.to(device)
        self.frame_sequence = self.frame_sequence.to(device)
        return self

    def copy(self):
        return ModelInputs(self.frame.clone(), self.frame_sequence.clone())
