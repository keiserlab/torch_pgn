import torch
from pgn.args import TrainArgs

class FPEncoder(torch.nn.Module):
    """Pass-through encoder module for fp datasets."""
    def __init__(self, args: TrainArgs):
        super(FPEncoder, self).__init__()

    def forward(self, data):
        return data.x