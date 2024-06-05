from typing import Callable, Union
from torch_pgn.models.dimenet_utils import *
from torch_pgn.args import TrainArgs

import torch

from torch_geometric.utils.scatter import scatter
from torch_geometric.typing import SparseTensor

from torch_geometric.nn.resolver import activation_resolver


class DimeNet(torch.nn.Module):
    r"""The directional message passing neural network (DimeNet) from the
    `"Directional Message Passing for Molecular Graphs"
    <https://arxiv.org/abs/2003.03123>`_ paper.
    DimeNet transforms messages based on the angle between them in a
    rotation-equivariant fashion.

    .. note::

        For an example of using a pretrained DimeNet variant, see
        `examples/qm9_pretrained_dimenet.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        qm9_pretrained_dimenet.py>`_.

    Args:
        hidden_channels (int): Hidden embedding size.
        out_channels (int): Size of each output sample.
        num_blocks (int): Number of building blocks.
        num_bilinear (int): Size of the bilinear layer tensor.
        num_spherical (int): Number of spherical harmonics.
        num_radial (int): Number of radial basis functions.
        cutoff (float, optional): Cutoff distance for interatomic
            interactions. (default: :obj:`5.0`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            collect for each node within the :attr:`cutoff` distance.
            (default: :obj:`32`)
        envelope_exponent (int, optional): Shape of the smooth cutoff.
            (default: :obj:`5`)
        num_before_skip (int, optional): Number of residual layers in the
            interaction blocks before the skip connection. (default: :obj:`1`)
        num_after_skip (int, optional): Number of residual layers in the
            interaction blocks after the skip connection. (default: :obj:`2`)
        num_output_layers (int, optional): Number of linear layers for the
            output blocks. (default: :obj:`3`)
        act (str or Callable, optional): The activation function.
            (default: :obj:`"swish"`)
    """

    url = ('https://github.com/klicperajo/dimenet/raw/master/pretrained/'
           'dimenet')

    def __init__(self, args: TrainArgs, atom_dim: int):

        super().__init__()

        if args.num_spherical < 2:
            raise ValueError("num_spherical should be greater than 1")

        act = activation_resolver(args.act)

        self.device = args.device
        self.cutoff = args.cutoff
        self.atom_dim = atom_dim
        self.max_num_neighbors = args.max_num_neighbors
        self.num_blocks = args.num_blocks
        self.num_radial = args.num_radial
        self.envelope_exponent = args.envelope_exponent
        self.num_spherical = args.num_spherical
        self.hidden_channels = args.nn_conv_internal_dim
        self.num_bilinear = args.num_bilinear
        self.num_before_skip = args.num_before_skip
        self.num_after_skip = args.num_after_skip
        #Maybe remove
        self.out_channels = args.out_channels
        self.num_output_layers = args.num_output_layers




        self.rbf = BesselBasisLayer(self.num_radial, self.cutoff, self.envelope_exponent).to(self.device)
        self.sbf = SphericalBasisLayer(self.num_spherical, self.num_radial, self.cutoff,
                                       self.envelope_exponent).to(self.device)

        self.emb = EmbeddingBlock(atom_dim, self.num_radial, self.hidden_channels, act).to(self.device)

        self.output_blocks = torch.nn.ModuleList([
            OutputBlock(self.num_radial, self.hidden_channels, self.out_channels,
                        self.num_output_layers, act).to(self.device) for _ in range(self.num_blocks + 1)
        ])

        self.interaction_blocks = torch.nn.ModuleList([
            InteractionBlock(self.hidden_channels, self.num_bilinear, self.num_spherical,
                             self.num_radial, self.num_before_skip, self.num_after_skip, act).to(self.device)
            for _ in range(self.num_blocks)
        ])

        # self.reset_parameters()

    def reset_parameters(self):
        self.rbf.reset_parameters()
        self.emb.reset_parameters()
        for out in self.output_blocks:
            out.reset_parameters()
        for interaction in self.interaction_blocks:
            interaction.reset_parameters()

    def triplets(self, edge_index, num_nodes):
        row, col = edge_index  # j->i

        value = torch.arange(row.size(0), device=row.device)  # .to(self.device)
        adj_t = SparseTensor(row=col, col=row, value=value,
                             sparse_sizes=(num_nodes, num_nodes))
        adj_t_row = adj_t[row]
        num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

        # Node indices (k->j->i) for triplets.
        idx_i = col.repeat_interleave(num_triplets)
        idx_j = row.repeat_interleave(num_triplets)
        idx_k = adj_t_row.storage.col()
        mask = idx_i != idx_k  # Remove i == k triplets.
        idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

        # Edge indices (k-j, j->i) for triplets.
        idx_kj = adj_t_row.storage.value()[mask]
        idx_ji = adj_t_row.storage.row()[mask]

        return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji

    def forward(self, node_feats, edge_index, pos, batch=None):
        """"""
        # edge_index = radius_graph(pos, r=self.cutoff, batch=batch,
        #                          max_num_neighbors=self.max_num_neighbors)
        # i -> dst_edge j-> src_edge

        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = self.triplets(
            edge_index, num_nodes=node_feats.size(0))

        # Calculate distances.
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

        # Calculate angles.
        pos_i = pos[idx_i]
        pos_ji, pos_ki = pos[idx_j] - pos_i, pos[idx_k] - pos_i
        a = (pos_ji * pos_ki).sum(dim=-1)
        b = torch.cross(pos_ji, pos_ki).norm(dim=-1)
        angle = torch.atan2(b, a)

        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)

        # Embedding block.
        x = self.emb(node_feats, rbf, i, j)

        # i -> dst edges
        P = self.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))
        # Interaction blocks.
        for interaction_block, output_block in zip(self.interaction_blocks,
                                                   self.output_blocks[1:]):
            x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
            P = P + output_block(x, rbf, i)

        return P.sum(dim=0) if batch is None else scatter(P, batch, dim=0)


class DimeNetPlusPlus(DimeNet):
    r"""The DimeNet++ from the `"Fast and Uncertainty-Aware
    Directional Message Passing for Non-Equilibrium Molecules"
    <https://arxiv.org/abs/2011.14115>`_ paper.

    :class:`DimeNetPlusPlus` is an upgrade to the :class:`DimeNet` model with
    8x faster and 10% more accurate than :class:`DimeNet`.

    Args:
        hidden_channels (int): Hidden embedding size.
        out_channels (int): Size of each output sample.
        num_blocks (int): Number of building blocks.
        int_emb_size (int): Size of embedding in the interaction block.
        basis_emb_size (int): Size of basis embedding in the interaction block.
        out_emb_channels (int): Size of embedding in the output block.
        num_spherical (int): Number of spherical harmonics.
        num_radial (int): Number of radial basis functions.
        cutoff: (float, optional): Cutoff distance for interatomic
            interactions. (default: :obj:`5.0`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            collect for each node within the :attr:`cutoff` distance.
            (default: :obj:`32`)
        envelope_exponent (int, optional): Shape of the smooth cutoff.
            (default: :obj:`5`)
        num_before_skip: (int, optional): Number of residual layers in the
            interaction blocks before the skip connection. (default: :obj:`1`)
        num_after_skip: (int, optional): Number of residual layers in the
            interaction blocks after the skip connection. (default: :obj:`2`)
        num_output_layers: (int, optional): Number of linear layers for the
            output blocks. (default: :obj:`3`)
        act: (str or Callable, optional): The activation funtion.
            (default: :obj:`"swish"`)
    """

    url = ('https://raw.githubusercontent.com/gasteigerjo/dimenet/'
           'master/pretrained/dimenet_pp')

    def __init__(self, args: TrainArgs, atom_dim: int):

        super().__init__(
            args=args,
            atom_dim=atom_dim
        )
        self.out_emb_channels = args.out_emb_channels
        self.int_emb_size = args.int_emb_size
        self.basis_emb_size = args.basis_emb_size

        act = activation_resolver(args.act)

        # We are re-using the RBF, SBF and embedding layers of `DimeNet` and
        # redefine output_block and interaction_block in DimeNet++.
        # Hence, it is to be noted that in the above initalization, the
        # variable `num_bilinear` does not have any purpose as it is used
        # solely in the `OutputBlock` of DimeNet:
        self.output_blocks = torch.nn.ModuleList([
            OutputPPBlock(self.num_radial, self.hidden_channels, self.out_emb_channels,
                          self.out_channels, self.num_output_layers, act).to(self.device)
            for _ in range(self.num_blocks + 1)
        ])

        self.interaction_blocks = torch.nn.ModuleList([
            InteractionPPBlock(self.hidden_channels, self.int_emb_size, self.basis_emb_size,
                               self.num_spherical, self.num_radial, self.num_before_skip,
                               self.num_after_skip, act).to(self.device) for _ in range(self.num_blocks)
        ])