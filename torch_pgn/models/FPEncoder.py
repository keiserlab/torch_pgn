import torch
from torch_pgn.args import TrainArgs

class FPEncoder(torch.nn.Module):
    """Pass-through encoder module for fp datasets."""
    def __init__(self, args: TrainArgs):
        super(FPEncoder, self).__init__()
        self.args = args
        self.fp_dim = args.fp_dim


    def forward(self, data):
        return data.x.reshape(data.num_graphs, self.fp_dim)