from pgn.graphs.graph_utils import (
    _renumber_nodes, _euclidean_distance,
    _convert_nodes_and_edges, _renumber_cross_edges,
    _get_atom_dict, _extract_position
)

from pgn.graphs.proximal_residues import *
from pgn.featurization.simple_featurization import *
from pgn.featurization.featurize import get_all_features

import networkx as nx
import matplotlib.pyplot as plt

#TODO: Add argument object for passing parameters other than ligand and protein

def yield_tree_reduction(ligand, protein, distance_cutoff=4.5, ignore_hoh=True, visualize=None, local_connect=True):
    """
    Takes in the receptor and the docked ligand and produces a tree version of this data. Calculated using Kruskal's and
    networkx.
    Note: This could likely be done much more efficiently, but not sure if this will be the bottleneck so can refactor
    if required.
    :param ligand: oddt.toolkit.Molecule object
            Molecules, which are analysed in order to find interactions.
    :param protein: oddt.toolkit.Molecule object
            Molecules, which are analysed in order to find interactions.
    :param distance_cutoff: float (default=4.5)
            Cutoff distance for close contacts.
    :param ignore_hoh: bool (default = True) Should the water molecules be ignored. This is based on the name of the
            residue ('HOH').
    :param visualize: whether to visualize the final output graph using matplotlib projection into 2d space
    :param local_connect: bool (default=True) Boolean flag to reestablish intra-ligand/protein connectivity after tree
    reduction.
    :return: A networkx graph of featurized nodes and edges representing the interaction of the protein and ligand.
    """
    protein_atoms, ligand_atoms = get_interacting_atoms(ligand, protein)
    cross_edges = extract_cross_edges(protein_atoms, ligand_atoms)
    receptor_nodes, receptor_edges = get_molecule_graph(protein, protein_atoms, depth=2)
    ligand_nodes, ligand_edges = get_molecule_graph(ligand, ligand_atoms, depth=1)

    ligand_dict, protein_dict = _renumber_nodes(ligand_nodes, receptor_nodes)

    if visualize is not None:
        ligand_positions_2d = _extract_position(ligand.atom_dict, ligand_nodes, ligand_dict, include_z=False)
        receptor_positions_2d = _extract_position(protein.atom_dict, receptor_nodes, protein_dict, include_z=False)
        pos2d = {**ligand_positions_2d, **receptor_positions_2d}

    ligand_positions_3d = _extract_position(ligand.atom_dict, ligand_nodes, ligand_dict, include_z=True)
    receptor_positions_3d = _extract_position(protein.atom_dict, receptor_nodes, protein_dict, include_z=True)
    pos3d = {**ligand_positions_3d, **receptor_positions_3d}

    atom_features, edges_features = get_all_features(ligand, protein, ligand_nodes, receptor_nodes,
                                                               ligand_edges, receptor_edges, cross_edges,
                                                               ligand_dict, protein_dict, pos3d)


    ligand_nodes, ligand_edges = _convert_nodes_and_edges(ligand_dict, ligand_nodes, ligand_edges)
    receptor_nodes, receptor_edges = _convert_nodes_and_edges(protein_dict, receptor_nodes, receptor_edges)
    cross_edges = _renumber_cross_edges(ligand_dict, protein_dict, cross_edges)

    G = nx.Graph()

    for node in ligand_nodes:
        G.add_node(node, atom_class='ligand', features=atom_features[node], pos3d=pos3d[node])
    for node in receptor_nodes:
        G.add_node(node, atom_class='receptor', features=atom_features[node], pos3d=pos3d[node])
    for idx, ligand_node in enumerate(cross_edges[0]):
        G.add_edge(ligand_node, cross_edges[1, idx], weight=_euclidean_distance(pos3d, ligand_node, cross_edges[1, idx]),
                   features=edges_features[(ligand_node, cross_edges[1, idx])])
    for idx, receptor_node in enumerate(receptor_edges[0]):
        G.add_edge(receptor_node, receptor_edges[1, idx],
                   weight=_euclidean_distance(pos3d, receptor_node, receptor_edges[1, idx]),
                   features=edges_features[(receptor_node, receptor_edges[1, idx])])
    for idx, ligand_node in enumerate(ligand_edges[0]):
        G.add_edge(ligand_node, ligand_edges[1, idx],
                   weight=_euclidean_distance(pos3d, ligand_node, ligand_edges[1, idx]),
                   features=edges_features[(ligand_node, ligand_edges[1, idx])])

    T = nx.minimum_spanning_tree(G)

    if local_connect:
        for idx, receptor_node in enumerate(receptor_edges[0]):
            T.add_edge(receptor_node, receptor_edges[1, idx],
                       weight=_euclidean_distance(pos3d, receptor_node, receptor_edges[1, idx]),
                       features=edges_features[(receptor_node, receptor_edges[1, idx])])
        for idx, ligand_node in enumerate(ligand_edges[0]):
            T.add_edge(ligand_node, ligand_edges[1, idx],
                       weight=_euclidean_distance(pos3d, ligand_node, ligand_edges[1, idx]),
                       features=edges_features[(ligand_node, ligand_edges[1, idx])])

    if visualize is not None:
        ec = nx.draw_networkx_edges(T, pos2d, alpha=0.6)
        nx.draw_networkx_nodes(T, pos2d,
                               nodelist=list(ligand_nodes),
                               node_color='r',
                               alpha=0.6)
        nx.draw_networkx_nodes(T, pos2d,
                               nodelist=list(receptor_nodes),
                               node_color='b',
                               alpha=0.6)
        plt.axis('off')
        if visualize is True:
            plt.show()
        else:
            plt.savefig(visualize)
        plt.clf()

    return T


def yield_full_interaction_graph(ligand, protein, distance_cutoff=4.5, ignore_hoh=True, visualize=None):
    """
    Takes in the receptor and the docked ligand and produces a graph data.
    :param ligand: oddt.toolkit.Molecule object
            Molecules, which are analysed in order to find interactions.
    :param protein: oddt.toolkit.Molecule object
            Molecules, which are analysed in order to find interactions.
    :param distance_cutoff: float (default=4.5)
            Cutoff distance for close contacts.
    :param ignore_hoh: bool (default = True) Should the water molecules be ignored. This is based on the name of the
            residue ('HOH').
    :param visualize: whether to visualize the final output graph using matplotlib projection into 2d space
    :return: A networkx graph of featurized nodes and edges representing the interaction of the protein and ligand.
    """
    protein_atoms, ligand_atoms = get_interacting_atoms(ligand, protein)
    cross_edges = extract_cross_edges(protein_atoms, ligand_atoms)
    receptor_nodes, receptor_edges = get_molecule_graph(protein, protein_atoms, depth=4)
    ligand_nodes, ligand_edges = get_molecule_graph(ligand, ligand_atoms, depth=2)

    ligand_dict, protein_dict = _renumber_nodes(ligand_nodes, receptor_nodes)


    if visualize is not None:
        ligand_positions_2d = _extract_position(ligand.atom_dict, ligand_nodes, ligand_dict, include_z=False)
        receptor_positions_2d = _extract_position(protein.atom_dict, receptor_nodes, protein_dict, include_z=False)
        pos2d = {**ligand_positions_2d, **receptor_positions_2d}

    ligand_positions_3d = _extract_position(ligand.atom_dict, ligand_nodes, ligand_dict, include_z=True)
    receptor_positions_3d = _extract_position(protein.atom_dict, receptor_nodes, protein_dict, include_z=True)
    pos3d = {**ligand_positions_3d, **receptor_positions_3d}

    atom_features, edges_features = get_all_features(ligand, protein, ligand_nodes, receptor_nodes,
                                                               ligand_edges, receptor_edges, cross_edges,
                                                               ligand_dict, protein_dict, pos3d)


    ligand_nodes, ligand_edges = _convert_nodes_and_edges(ligand_dict, ligand_nodes, ligand_edges)
    receptor_nodes, receptor_edges = _convert_nodes_and_edges(protein_dict, receptor_nodes, receptor_edges)
    cross_edges = _renumber_cross_edges(ligand_dict, protein_dict, cross_edges)

    G = nx.Graph()

    for node in ligand_nodes:
        G.add_node(node, atom_class='ligand', features=atom_features[node], pos3d=pos3d[node])
    for node in receptor_nodes:
        G.add_node(node, atom_class='receptor', features=atom_features[node], pos3d=pos3d[node])
    for idx, ligand_node in enumerate(cross_edges[0]):
        G.add_edge(ligand_node, cross_edges[1, idx], weight=_euclidean_distance(pos3d, ligand_node, cross_edges[1, idx]),
                   features=edges_features[(ligand_node, cross_edges[1, idx])])
    for idx, receptor_node in enumerate(receptor_edges[0]):
        G.add_edge(receptor_node, receptor_edges[1, idx],
                   weight=_euclidean_distance(pos3d, receptor_node, receptor_edges[1, idx]),
                   features=edges_features[(receptor_node, receptor_edges[1, idx])])
    for idx, ligand_node in enumerate(ligand_edges[0]):
        G.add_edge(ligand_node, ligand_edges[1, idx],
                   weight=_euclidean_distance(pos3d, ligand_node, ligand_edges[1, idx]),
                   features=edges_features[(ligand_node, ligand_edges[1, idx])])

    if visualize is not None:
        ec = nx.draw_networkx_edges(G, pos2d, alpha=0.6)
        nx.draw_networkx_nodes(G, pos2d,
                               nodelist=list(ligand_nodes),
                               node_color='r',
                               alpha=0.6)
        nx.draw_networkx_nodes(G, pos2d,
                               nodelist=list(receptor_nodes),
                               node_color='b',
                               alpha=0.6)
        plt.axis('off')
        if visualize is True:
            plt.show()
        else:
            plt.savefig(visualize)
        plt.clf()

    return G
