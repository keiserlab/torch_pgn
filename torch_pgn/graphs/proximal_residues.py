import numpy as np

import oddt
from oddt.utils import is_openbabel_molecule
from oddt.interactions import close_contacts


"""
The code in this module is mostly modified versions of the ODDT implementations of PLEC and ECFP/Morgan.
"""


def get_interacting_atoms(ligand, protein, distance_cutoff=4.5, ignore_hoh=True):
    """
    Protein ligand interacting atom extractor. Takes a ligand and protein pair and uses the built in ODDT functions
    to output the interacting residue.
    NOTE: The exact format needs to be decided on in order to be as general as possible. Will start with just being a
    helped function and the exact form can be finalized as the MPNN frameworks become more clear.
    :param ligand: oddt.toolkit.Molecule object
            Molecules, which are analysed in order to find interactions.
    :param protein: oddt.toolkit.Molecule object
            Molecules, which are analysed in order to find interactions.
    :param distance_cutoff: float (default=4.5)
            Cutoff distance for close contacts.
    :param ignore_hoh: bool (default = True) Should the water molecules be ignored. This is based on the name of the
            residue ('HOH').
    :return: ???? (Probably start with the interacting residues for testing and then move on for there (Need to figure
     this out.)
    """
    # removing h
    protein_mask = protein_no_h = (protein.atom_dict['atomicnum'] != 1)
    if ignore_hoh:
        # a copy is needed, so not modifing inplace
        protein_mask = protein_mask & (protein.atom_dict['resname'] != 'HOH')
    protein_dict = protein.atom_dict[protein_mask]
    ligand_dict = ligand.atom_dict[ligand.atom_dict['atomicnum'] != 1]

    # atoms in contact
    protein_atoms, ligand_atoms = close_contacts(
        protein_dict, ligand_dict, cutoff=distance_cutoff)

    return protein_atoms, ligand_atoms


def get_proximal_atoms(ligand, protein, distance_cutoff=4.5, ignore_hoh=True):
    """
    Protein ligand interacting atom extractor. Takes a ligand and protein pair and uses the built in ODDT functions
    to output the interacting residue.
    NOTE: The exact format needs to be decided on in order to be as general as possible. Will start with just being a
    helped function and the exact form can be finalized as the MPNN frameworks become more clear.
    :param ligand: oddt.toolkit.Molecule object
            Molecules, which are analysed in order to find interactions.
    :param protein: oddt.toolkit.Molecule object
            Molecules, which are analysed in order to find interactions.
    :param distance_cutoff: float (default=4.5)
            Cutoff distance for close contacts.
    :param ignore_hoh: bool (default = True) Should the water molecules be ignored. This is based on the name of the
            residue ('HOH').
    :return: ???? (Probably start with the interacting residues for testing and then move on for there (Need to figure
     this out.)
    """
    #TODO: Allow toggle for hydrogen and water enables by args
    # removing h
    protein_mask = protein_no_h = (protein.atom_dict['atomicnum'] != 1)
    if ignore_hoh:
        # a copy is needed, so not modifing inplace
        protein_mask = protein_mask & (protein.atom_dict['resname'] != 'HOH')
    protein_dict = protein.atom_dict[protein_mask]
    ligand_dict = ligand.atom_dict[ligand.atom_dict['atomicnum'] != 1]

    # atoms in contact
    protein_atoms, ligand_atoms = close_contacts(
        protein_dict, ligand_dict, cutoff=distance_cutoff)

    return protein_atoms, ligand_atoms


def extract_cross_edges(protein_atoms, ligand_atoms):
    """
    Creates a set of cross edges formatted in pytorch geometric compatible format.
    :param protein_atoms: Numpy atom_dict of the form output by get_interacting atoms (see close_contract documentation
    for ODDT: https://oddt.readthedocs.io/en/0.1.1/rst/oddt.html) for the interaction protein atoms
    :param ligand_atoms: Numpy atom_dict of the form output by get_interacting atoms (see close_contract documentation
    for ODDT: https://oddt.readthedocs.io/en/0.1.1/rst/oddt.html) for the interaction ligand atoms
    :return: A numpy array of edge formatted for pytorch. Format is [[s1, s2, s3...], [d1, d2, d3, ...]].
    All source vertices will be in the ligand and all dest vectices will be in protein. Final processing will be done
    in order to make graph undirected if desired. And have correct numbering
    """
    lig_labels = ligand_atoms['id'].tolist()
    protein_labels = protein_atoms['id'].tolist()
    return np.array([lig_labels, protein_labels])


def get_molecule_graph(molecule, molecule_atoms, depth=1):
    """
    Constructs the ligand portion of the interaction graph (intra-ligand edges and nodes) and returns them as a set of
    node index values and an edge array.
    :param molecule: oddt.toolkit.Molecule object
            Molecule, which are analysed in order to find interactions.
    :param molecule_atoms: The molecule atoms involved in an interaction with the interaction partner formatted as a
    numpy atom_dict (See https://oddt.readthedocs.io/en/0.1.1/rst/oddt.html)
    :param depth: The depth away from interacting atoms to go in the molecular graph to construct the final
    interaction graph. i.e. if molecule atom 5 is interacting and has neighbors 3, and 4 that don't interact, if the
    depth is == 1 then 3 and 4 will not be included in the interaction graph if depth is == 2 then both 3 and 4 will be
    added to the interaction graph. For depth > 2 this process is repeated iteratively for the new atoms added to the
    interaction graph. If depth == 0 no intra-molecule edges will be added. (This may be a poor initial choice, but
    it seems like this may be the best approximation to PLEC)
    :return: nodes, edges:
        nodes = a set() of nodes to be included in the interaction graph.
        edges = A numpy array of edge formatted for pytorch. Format is [[s1, s2, s3...], [d1, d2, d3, ...]].
        All edges will be bidirectional i.e. iff i -> j exists j -> i exists.
    """
    node_set = set(molecule_atoms['id'].tolist())
    neighbor_nodes = set()
    if depth == 0:
        return node_set, np.array([], [])
    edges = set()
    for idx in node_set:
        if is_openbabel_molecule(molecule):
            envs = [[idx]]
            visited = [idx]
            for r in range(1, depth + 1):
                tmp = []
                for atom_idx in envs[r - 1]:
                    for neighbor in oddt.toolkits.ob.ob.OBAtomAtomIter(molecule.OBMol.GetAtom(atom_idx + 1)):
                        if neighbor.GetAtomicNum() == 1:
                            continue
                        n_idx = neighbor.GetIdx() - 1
                        if n_idx not in visited and n_idx not in tmp:
                            tmp.append(n_idx)
                            visited.append(n_idx)
                            neighbor_nodes.add(n_idx)
                            edges.add((atom_idx, n_idx))
                            edges.add((n_idx, atom_idx))
                envs.append(tmp)
        else:
            envs = [[idx]]
            visited = [idx]
            for r in range(1, depth + 1):
                tmp = []
                for atom_idx in envs[r - 1]:
                    for neighbor in molecule.Mol.GetAtomWithIdx(atom_idx).GetNeighbors():
                        if neighbor.GetAtomicNum() == 1:
                            continue
                        n_idx = neighbor.GetIdx()
                        if n_idx not in visited and n_idx not in tmp:
                            tmp.append(n_idx)
                            visited.append(n_idx)
                            neighbor_nodes.add(n_idx)
                            edges.add((atom_idx, n_idx))
                            edges.add((n_idx, atom_idx))
                envs.append(tmp)

    final_edges = np.array(list(edges)).T
    if final_edges.size == 0:
        final_edges = np.array([[],[]])
    return node_set.union(neighbor_nodes), final_edges


