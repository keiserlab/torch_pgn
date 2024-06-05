import numpy as np

import oddt
from oddt.utils import is_openbabel_molecule
from oddt.interactions import close_contacts
from oddt.fingerprints import _ECFP_atom_repr

import torch_pgn.graphs.graph_utils as gu

"""
Functions to output basic atom representations.
"""

def featurize_atoms_ECFP_like(molecule, index_list, is_ligand, translate=None):
    """
    Outputs a list of reps for each atom in the molecule that is contained in the index list.
    :param molecule: oddt.toolkit.Molecule object
            Molecules, which is analysed in order to calculate representations
    :param index_list: List of atom numbers to extract features
    :param is_ligand: (bool) is in ligand
    :param translate: (dict) If the nodes have already been translated then this will allow for going back to previous
            rep.
    :return: a dictionary with the key, value pairs corresponding to node, ECFP_like rep.
    The default output features are:
     (atomic number (int), isotope_id, total degree (excl_Hs), N_hydrogens, formal charge, ring, is aromatic, ligand (1)
     protein(0)
    """
    molecule_atom_repr = {}
    for aidx in index_list:
        ECFP = np.array(list(_ECFP_atom_repr(molecule, aidx)) + [int(is_ligand)])
        if translate is None:
            molecule_atom_repr[aidx] = ECFP
        else:
            molecule_atom_repr[translate[aidx]] = ECFP

    return molecule_atom_repr


def featurize_edges_simple(bond_list, molecule, translate, position_dict):
    """
    A typesafe way of retrieving bond information from ODDT molecule objects or interaction_edges
    :param bond_list: a list of bonds in (u, v) format
    :param molecule: molecule of type oddt.toolkit.Molecule to retrive bonds from
    :param translate: Translation dictionary to store the nodes in
    :return: A dictionary of bond objects for further processing. Each bond will be stored in both (u, v) and (v, u)
    regardless of whether both are passed in the bond_list
    Note: OB bond docs: http://openbabel.org/dev-api/classOpenBabel_1_1OBBond.shtml#a892ffd8f4ddd7adbe285b6bf7133aa7a
    Note: Rdkit bond docs: https://www.rdkit.org/docs/cppapi/classRDKit_1_1Bond.html
    """
    bond_dict = {}
    if is_openbabel_molecule(molecule):
        get_bond = lambda u, v: molecule.OBMol.GetBond(int(u + 1),  int(v + 1))
    elif molecule is not None:
        get_bond = lambda u, v: molecule.Mol.GetBond(u, v)
    else:
        get_bond = lambda u, v: None
    for bond in bond_list:
        u, v = bond
        bond_dict[(u, v)] = get_bond(u, v)
    feature_dict = {}
    if is_openbabel_molecule(molecule):
        for u, v in bond_dict.keys():
            bond = bond_dict[(u, v)]
            order = bond.GetBondOrder()
            feature = np.array([gu._distance(position_dict, translate[u], translate[v]),
                                int(order == 1), int(order == 2), int(order == 3),
                                int(bond.IsAromatic()), int(molecule is None)])
            feature_dict[(translate[u], translate[v])]= feature
            feature_dict[(translate[v], translate[u])] = feature
    elif molecule is not None:
        for u, v in bond_dict.keys():
            bond = bond_dict[(u, v)]
            feature = np.array([gu._distance(position_dict, translate[u], translate[v]),
                                int(bond.getIsSingle()), int(bond.getIsDouble()), int(bond.getIsTriple()),
                                int(bond.getIsAromatic()), int(molecule is None)])
            feature_dict[(translate[u], translate[v])] = feature
            feature_dict[(translate[v], translate[u])] = feature
    else:
        ligand_translate, protein_translate = translate
        for u, v in bond_dict.keys():
            feature = np.array(
                [gu._distance(position_dict, ligand_translate[u], protein_translate[v]),
                 0, 0, 0, 0, int(molecule is None)])
            feature_dict[(ligand_translate[u], protein_translate[v])] = feature
            feature_dict[(protein_translate[v], ligand_translate[u])] = feature
    return feature_dict
