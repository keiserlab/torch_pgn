from torch_pgn.graphs.proximity_graph import (yield_full_interaction_graph, yield_tree_reduction)

from oddt.utils import is_openbabel_molecule

import networkx as nx
import numpy as np

import pickle

"""Utility functions used in graph construction."""

def _renumber_nodes(ligand_nodes, protein_nodes):
    """
    Takes the ligand nodes and protein nodes and renumbers them base zero to comply with the PyTorch convention for
    input working_data. The ligand nodes will start at zero and the receptor nodes begin numbering thereafter.
    :param ligand_nodes: A set of nodes in the ligand.
    :param protein_nodes: A set of node in the receptor.
    :return: A ligand dictionary and a receptor dictionary that takes in the old number and outputs the safe node numbering
    as well as the offset for converting protein nodes back to pdb numbering.
    """
    ligand_dict = {idx: i for i, idx in enumerate(ligand_nodes)}
    offset = len(ligand_dict.keys())
    protein_dict = {idx: i + offset for i, idx in enumerate(protein_nodes)}
    return ligand_dict, protein_dict, offset


def _distance(pos, idx1, idx2, norm_type='l1'):
    """
    Get the euclidean distance between two nodes
    :param pos: position dictionary indexed by node
    :param idx1: index of node 1 in pos dictionary
    :param idx2: index of node 2 in pos dictionary
    :return: euclidean distance (float)
    """
    if norm_type == 'l1':
        dist = np.sqrt(np.sum(np.square((pos[idx1] - pos[idx2]))))
    elif norm_type == 'l2':
        dist = np.sum(np.sqrt(np.square((pos[idx1] - pos[idx2]))))
    else:
        raise ValueError("norm_type must be either l1 or l2.")
    return dist


def _convert_nodes_and_edges(translate, nodes, edges):
    """
    Ensures proper numbering of the nodes and edges in the Proximity Graph
    :param translate: dictionary used to translate between arbitrary numbering to 0 based numbering
    :param nodes: noes of the proximity graph
    :param edges: edges of the proximity graph
    :return: Tuple(Set(nodes), edges as a numpy array [[u1, u2, u3,....][v1, v2, v3...]]
    """
    translated_nodes = set()
    for node in nodes:
        translated_nodes.add(translate[node])
    if edges.size != 0:
        for i in range(edges.shape[1]):
            edges[0, i] = translate[edges[0, i]]
            edges[1, i] = translate[edges[1, i]]
    return translated_nodes, edges


def _renumber_cross_edges(ligand_translator, protein_translator, cross_edges):
    """
    Assumes ligand nodes are in the cross_edges[0] array and protein nodes are in the cross_edges[1] array.
    :return: Properly renumbered edges.
    """
    for i in range(len(cross_edges[0])):
        cross_edges[0, i] = ligand_translator[cross_edges[0, i]]
        cross_edges[1, i] = protein_translator[cross_edges[1, i]]
    return cross_edges


def _get_atom_dict(node_list, molecule, translate):
    """
    Returns the atom objects of the type corresponding to the molecule for further use
    :param node_list: List of node numbers to retrieve
    :param molecule: moleucule of type oddt.toolkit.Molecule to retrive atoms from
    :param translate: Translation dictionary to store the nodes in
    :return: A dictionary that maps the translated node/atoms number to the underlying atom object
    """
    atom_dict = {}
    if is_openbabel_molecule(molecule):
        get_atom = lambda atom_idx: molecule.OBMol.GetAtom(atom_idx + 1)
    else:
        get_atom = lambda atom_idx: molecule.Mol.GetAtomWithIdx(atom_idx)
    for node in node_list:
        atom_dict[translate[node]] = get_atom(node)
    return atom_dict


def _extract_position(molecule_dict, atom_list, translate, include_z=False):
    """
    Simple function to extract the position of an atom from the molecule dict and return a dictionary with a position.
    :param molecule_dict: molecule dictionary from oddt
    :param atom_list: list of atoms to extract the position of (as node index)
    :param translate: dictionary used to translate from the arbitrary node numbering from the parent structure to the
    logical 0 based numbering used in the final proximity graph
    :return: A node indexed dictionary of node positions.
    """
    position_dict = {}
    for atom in atom_list:
        try:
            x, y, z = molecule_dict[molecule_dict['id'] == atom]['coords'][0]
            if include_z:
                position_dict[translate[atom]] = np.array((x,y,z))
            else:
                position_dict[translate[atom]] = np.array((x,y))
        except:
            if include_z:
                position_dict[translate[atom]] = np.array((-1, -1, -1))
            else:
                position_dict[translate[atom]] = np.array((-1, -1))
    return position_dict


def _return_graph(input_tuple):
    """
    Return the proximity graph resulting from a given receptor molecule pair.
    :param protein: The receptor structure for given interaction graph TODO: Type...also fix oddt
    :param ligand: The molecule (as oddt pandas object)
    :param energy:The ground truth energy of this interaction
    :param name:The name of this proximity graph
    :param args: Arguments object used to determine the setting used for graph generation
    :return: Returns the name, the graph (networkx), and energy of the proximity graph
    """
    #TODO: make interaction graph yielding functions compatible with args object
    protein, ligand, energy, name, proximity_radius, ligand_depth, receptor_depth = input_tuple
    mode = 'full'
    if mode == 'full':
        graph = yield_full_interaction_graph(ligand, protein, name, distance_cutoff=proximity_radius, lig_depth=ligand_depth,
                                     receptor_depth=receptor_depth)
    elif mode == 'tree':
        graph = yield_tree_reduction(ligand, protein, name, distance_cutoff=proximity_radius, lig_depth=ligand_depth,
                                     receptor_depth=receptor_depth)
    else:
        raise ValueError("Illegal graph_type argument. Please choose from <full, tree>")
    return name, graph, energy
