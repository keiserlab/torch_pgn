import torch_pgn.graphs.graph_utils as gu

from torch_pgn.graphs.proximal_residues import *
from torch_pgn.featurization.simple_featurization import *
from torch_pgn.featurization.featurize import get_all_features

import networkx as nx
import matplotlib.pyplot as plt
import os.path as osp

#TODO: Add argument object for passing parameters other than ligand and protein

def yield_tree_reduction(ligand, protein, name, distance_cutoff=4.5, lig_depth=2, receptor_depth=4,
                                 ignore_hoh=True, visualize=False, local_connect=True):
    """
    Takes in the receptor and the docked ligand and produces a tree version of this working_data. Calculated using Kruskal's and
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
    protein_atoms, ligand_atoms = get_interacting_atoms(ligand, protein, distance_cutoff=distance_cutoff)
    cross_edges = extract_cross_edges(protein_atoms, ligand_atoms)
    if lig_depth == -1:
        #Keep all ligand edges as opposed to the proximal and add depth handling
        ligand_nodes, ligand_edges = get_molecule_graph(ligand, ligand.atom_dict[ligand.atom_dict['atomicnum'] != 1], depth=2)
    else:
        ligand_nodes, ligand_edges = get_molecule_graph(ligand, ligand_atoms, depth=lig_depth)
    receptor_nodes, receptor_edges = get_molecule_graph(protein, protein_atoms, depth=receptor_depth)

    ligand_dict, protein_dict, offset = gu._renumber_nodes(ligand_nodes, receptor_nodes)

    if visualize is not None:
        ligand_positions_2d = gu._extract_position(ligand.atom_dict, ligand_nodes, ligand_dict, include_z=False)
        receptor_positions_2d = gu._extract_position(protein.atom_dict, receptor_nodes, protein_dict, include_z=False)
        pos2d = {**ligand_positions_2d, **receptor_positions_2d}

    ligand_positions_3d = gu._extract_position(ligand.atom_dict, ligand_nodes, ligand_dict, include_z=True)
    receptor_positions_3d = gu._extract_position(protein.atom_dict, receptor_nodes, protein_dict, include_z=True)
    pos3d = {**ligand_positions_3d, **receptor_positions_3d}

    atom_features, edges_features = get_all_features(ligand, protein, ligand_nodes, receptor_nodes,
                                                               ligand_edges, receptor_edges, cross_edges,
                                                               ligand_dict, protein_dict, pos3d)


    ligand_nodes, ligand_edges = gu._convert_nodes_and_edges(ligand_dict, ligand_nodes, ligand_edges)
    receptor_nodes, receptor_edges = gu._convert_nodes_and_edges(protein_dict, receptor_nodes, receptor_edges)
    cross_edges = gu._renumber_cross_edges(ligand_dict, protein_dict, cross_edges)

    G = nx.Graph()
    pdb_idx = {v: k for k, v in protein_dict.items()}

    for node in ligand_nodes:
        G.add_node(node, atom_class='ligand', features=atom_features[node], pos3d=pos3d[node])
    for node in receptor_nodes:
        G.add_node(node, atom_class='receptor', features=atom_features[node], pos3d=pos3d[node], pdb_idx=pdb_idx[node])
    for idx, ligand_node in enumerate(cross_edges[0]):
        G.add_edge(ligand_node, cross_edges[1, idx], weight=gu._distance(pos3d, ligand_node, cross_edges[1, idx]),
                   features=edges_features[(ligand_node, cross_edges[1, idx])])
    for idx, receptor_node in enumerate(receptor_edges[0]):
        G.add_edge(receptor_node, receptor_edges[1, idx],
                   weight=gu._distance(pos3d, receptor_node, receptor_edges[1, idx]),
                   features=edges_features[(receptor_node, receptor_edges[1, idx])])
    for idx, ligand_node in enumerate(ligand_edges[0]):
        G.add_edge(ligand_node, ligand_edges[1, idx],
                   weight=gu._distance(pos3d, ligand_node, ligand_edges[1, idx]),
                   features=edges_features[(ligand_node, ligand_edges[1, idx])])

    G.name = name
    T = nx.minimum_spanning_tree(G)

    if local_connect:
        for idx, receptor_node in enumerate(receptor_edges[0]):
            T.add_edge(receptor_node, receptor_edges[1, idx],
                       weight=gu._distance(pos3d, receptor_node, receptor_edges[1, idx]),
                       features=edges_features[(receptor_node, receptor_edges[1, idx])])
        for idx, ligand_node in enumerate(ligand_edges[0]):
            T.add_edge(ligand_node, ligand_edges[1, idx],
                       weight=gu._distance(pos3d, ligand_node, ligand_edges[1, idx]),
                       features=edges_features[(ligand_node, ligand_edges[1, idx])])

    if visualize is not None:
        ec = nx.draw_networkx_edges(T, pos2d, alpha=0.6)
        nx.draw_networkx_nodes(T, pos2d,
                               nodelist=list(ligand_nodes),
                               node_color='r',
                               alpha=0.6)
        if len(receptor_nodes) > 0:
            nx.draw_networkx_nodes(T, pos2d,
                                   nodelist=list(receptor_nodes),
                                   node_color='b',
                                   alpha=0.6)
        plt.axis('off')
        if visualize is True:
            plt.title(G.name)
            plt.show()
        else:
            plt.savefig(visualize)
        plt.clf()

    return T


def yield_full_interaction_graph(ligand, protein, name, distance_cutoff=4.5, lig_depth=2, receptor_depth=4,
                                 ignore_hoh=True, visualize=False):
    """
    Takes in the receptor and the docked ligand and produces a graph working_data.
    :param ligand: oddt.toolkit.Molecule object
            Molecules, which are analysed in order to find interactions.
    :param protein: oddt.toolkit.Molecule object
            Molecules, which are analysed in order to find interactions.
    :param distance_cutoff: float (default=4.5)
            Cutoff distance for close contacts.
    :param ignore_hoh: bool (default = True) Should the water molecules be ignored. This is based on the name of the
            residue ('HOH').
    :param visualize: whether to visualize the final output graph using matplotlib projection into 2d space
    :param maintain_lig: If True the entire ligand will be maintained regardless of atom proximity. Otherwise the ligand
    will be allowed to fragment based on the proximity graph.
    :return: A networkx graph of featurized nodes and edges representing the interaction of the protein and ligand.
    """
    protein_atoms, ligand_atoms = get_interacting_atoms(ligand, protein, distance_cutoff=distance_cutoff)
    cross_edges = extract_cross_edges(protein_atoms, ligand_atoms)
    receptor_nodes, receptor_edges = get_molecule_graph(protein, protein_atoms, depth=receptor_depth)
    if lig_depth == -1:
        #Keep all ligand edges as opposed to the proximal and add depth handling
        ligand_nodes, ligand_edges = get_molecule_graph(ligand, ligand.atom_dict[ligand.atom_dict['atomicnum'] != 1], depth=2)
    else:
        ligand_nodes, ligand_edges = get_molecule_graph(ligand, ligand_atoms, depth=lig_depth)

    ligand_dict, protein_dict, offset = gu._renumber_nodes(ligand_nodes, receptor_nodes)


    if visualize is not None:
        ligand_positions_2d = gu._extract_position(ligand.atom_dict, ligand_nodes, ligand_dict, include_z=False)
        receptor_positions_2d = gu._extract_position(protein.atom_dict, receptor_nodes, protein_dict, include_z=False)
        pos2d = {**ligand_positions_2d, **receptor_positions_2d}

    ligand_positions_3d = gu._extract_position(ligand.atom_dict, ligand_nodes, ligand_dict, include_z=True)
    receptor_positions_3d = gu._extract_position(protein.atom_dict, receptor_nodes, protein_dict, include_z=True)
    pos3d = {**ligand_positions_3d, **receptor_positions_3d}

    atom_features, edges_features = get_all_features(ligand, protein, ligand_nodes, receptor_nodes,
                                                               ligand_edges, receptor_edges, cross_edges,
                                                               ligand_dict, protein_dict, pos3d)


    ligand_nodes, ligand_edges = gu._convert_nodes_and_edges(ligand_dict, ligand_nodes, ligand_edges)
    receptor_nodes, receptor_edges = gu._convert_nodes_and_edges(protein_dict, receptor_nodes, receptor_edges)
    cross_edges = gu._renumber_cross_edges(ligand_dict, protein_dict, cross_edges)

    G = nx.Graph()
    G.name = name

    pdb_idx = {v: k for k, v in protein_dict.items()}

    for node in ligand_nodes:
        G.add_node(node, atom_class='ligand', features=atom_features[node], pos3d=pos3d[node])
    for node in receptor_nodes:
        G.add_node(node, atom_class='receptor', features=atom_features[node], pos3d=pos3d[node], pdb_idx=pdb_idx[node])
    for idx, ligand_node in enumerate(cross_edges[0]):
        G.add_edge(ligand_node, cross_edges[1, idx], weight=gu._distance(pos3d, ligand_node, cross_edges[1, idx]),
                   features=edges_features[(ligand_node, cross_edges[1, idx])])
    for idx, receptor_node in enumerate(receptor_edges[0]):
        G.add_edge(receptor_node, receptor_edges[1, idx],
                   weight=gu._distance(pos3d, receptor_node, receptor_edges[1, idx]),
                   features=edges_features[(receptor_node, receptor_edges[1, idx])])
    for idx, ligand_node in enumerate(ligand_edges[0]):
        G.add_edge(ligand_node, ligand_edges[1, idx],
                   weight=gu._distance(pos3d, ligand_node, ligand_edges[1, idx]),
                   features=edges_features[(ligand_node, ligand_edges[1, idx])])

    #TODO: Make visualize a helper function and make it work with args object
    if visualize is not None:
        ec = nx.draw_networkx_edges(G, pos2d, alpha=0.6)
        nx.draw_networkx_nodes(G, pos2d,
                               nodelist=list(ligand_nodes),
                               node_color='r',
                               alpha=0.6)
        if len(receptor_nodes) > 0:
            nx.draw_networkx_nodes(G, pos2d,
                                   nodelist=list(receptor_nodes),
                                   node_color='b',
                                   alpha=0.6)
        plt.axis('off')
        if visualize:
            plt.title(G.name)
            plt.show()
        plt.clf()

    return G
