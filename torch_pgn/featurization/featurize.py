from torch_pgn.featurization.simple_featurization import (
    featurize_atoms_ECFP_like, featurize_edges_simple
)

######################### Function to calculated all of the atom and edge features #####################################

#TODO: make featurization function an argument to allow for other possible choices (low priority)

def get_all_features(ligand, protein, ligand_nodes, protein_nodes, ligand_edges, protein_edges, cross_edges,
                     ligand_translate, protein_translate, positions):
    """
    Takes in all the graph information and returns the features as defined in featurize_atoms_ECFP_like and
    featurize_edges simple.
    :param ligand: molecule of type oddt.toolkit.Molecule corresponsiding to ligand
    :param protein: molecule of type oddt.toolkit.Molecule corresponding to protein
    :param ligand_nodes: set of nodes in the ligand molecule
    :param protein_nodes: set of nodes in the protein molecule
    :param ligand_edges: set of edges existing only between ligand atoms
    :param protein_edges: set of edges existing only between protein atoms
    :param cross_edges: set of edges with one end in ligand and the other in protein. These are the so-called interaction
    edges.
    :param ligand_translate: A dictionary to renumber the ligand nodes (required to guarantee that node numberings will
    not collide between the ligand and protein nodes.)
    :param protein_translate: A dictionary to renumber the protein nodes (required to guarantee that node numberings will
    not collide between the ligand and protein nodes.)
    :return: A dictionary with atomwise features indexed by node number and a dictionary of edgewise feature indexed by
    (u, v) tuples.
    """
    ligand_atom_feature = featurize_atoms_ECFP_like(ligand, ligand_nodes, True, ligand_translate)
    protein_atom_feature = featurize_atoms_ECFP_like(protein, protein_nodes, False, protein_translate)
    if len(ligand_edges) > 0:
        ligand_edge_feature = featurize_edges_simple(set(zip(ligand_edges[0], ligand_edges[1])), ligand, ligand_translate, positions)
    else:
        ligand_edge_feature = {}
    if len(protein_edges) > 0:
        protein_edge_feature = featurize_edges_simple(set(zip(protein_edges[0], protein_edges[1])), protein, protein_translate, positions)
    else:
        protein_edge_feature = {}
    if len(cross_edges) > 0:
        cross_edge_feature = featurize_edges_simple(set(zip(cross_edges[0], cross_edges[1])), None, [ligand_translate, protein_translate], positions)
    else:
        cross_edge_feature = {}
    atom_features = {**ligand_atom_feature, **protein_atom_feature}
    edge_features = {**ligand_edge_feature, **protein_edge_feature, **cross_edge_feature}
    return atom_features, edge_features
