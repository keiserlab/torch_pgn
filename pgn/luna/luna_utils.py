# LUNA utils to calculate interactions

# LUNA imports (LUNA version atw: 0.12.0)
from luna.projects import StructureCache 
from luna.mol.entry import MolFileEntry
from luna.mol.features import FeatureExtractor
from luna.mol.groups import AtomGroupPerceiver
from luna.config.params import ProjectParams
from luna.interaction.contact import get_contacts_with
from luna.interaction.calc import InteractionCalculator, InteractionsManager
from luna.interaction.filter import InteractionFilter, BindingModeFilter
from luna.interaction.fp.shell import ShellGenerator
from luna.interaction.fp.type import IFPType
from luna.util.default_values import *
from luna.MyBio.util import get_entity_from_entry
from luna.MyBio.PDB.PDBParser import PDBParser

from rdkit.Chem import ChemicalFeatures

import os

# functions modified from luna/projects.py
def calculate_interactions(prot_id, zid, loaded_mol, pdb_file,
                           cache = None,
                           add_proximal = True):
    
    # read in protein/mol, get covalent bonds + atomic groups
    entry = MolFileEntry.from_mol_obj(prot_id, zid,
                    loaded_mol)
    
    entry.pdb_file = pdb_file

    pdb_parser = PDBParser(PERMISSIVE=True, QUIET=True,
                                   FIX_EMPTY_CHAINS=True,
                                   FIX_ATOM_NAME_CONFLICT=True,
                                   FIX_OBABEL_FLAGS=False)
    
    #update structure - CURRENTLY WORKS FOR MOLFILEENTRY ONLY
    structure = pdb_parser.get_structure(entry.pdb_id, pdb_file)
    structure = entry.get_biopython_structure(structure, pdb_parser)
    add_h = _decide_hydrogen_addition(True, pdb_parser.get_header(), entry)
    
    ligand = get_entity_from_entry(structure, entry)
    ligand.set_as_target(is_target = True)
    
    # TODO: cacheing - should we return the cache as well?
    atm_grps_mngr = _perceive_chemical_groups(entry,
                                            structure[0],
                                            ligand, add_h,
                                            cache=cache)
    
    # determine non-covalent interactions - at this point just PLIs, no self-self
    atm_grps_mngr, interactions_mngr = _perceive_noncovalent_interactions(entry, atm_grps_mngr, 
                                                    add_proximal = add_proximal)
    return atm_grps_mngr, interactions_mngr


def _perceive_noncovalent_interactions(entry, atm_grps_mngr, binding_mode_filter = None, 
                                                      add_proximal = True):
    pli_filter = InteractionFilter.new_pli_filter(ignore_self_inter = True, 
                                                  ignore_any_h2o = True,
                                                 ignore_hetatm_hetatm = True, 
                                                 ignore_res_res = True)
    inter_calc = InteractionCalculator(inter_filter=pli_filter, 
                                               strict_donor_rules=True, 
                                               add_proximal = add_proximal)
    
    interactions_mngr = inter_calc.calc_interactions(atm_grps_mngr.atm_grps)
    interactions_mngr.entry = entry
    
    atm_grps_mngr.merge_hydrophobic_atoms(interactions_mngr)
        
    return atm_grps_mngr, interactions_mngr


def _decide_hydrogen_addition(try_h_addition, pdb_header, entry):
    if try_h_addition:
        if "structure_method" in pdb_header:
            method = pdb_header["structure_method"]
            # If the method is not a NMR type does not add hydrogen as it usually already has hydrogens.
            if method.upper() in NMR_METHODS:
                return False
        return True
    return False

def _perceive_chemical_groups(entry, entity, ligand,
                                  add_h=False, cache=None):

        radius = BOUNDARY_CONFIG["bsite_cutoff"]
        nb_pairs = get_contacts_with(entity, ligand, level='R', radius=radius)

        mol_objs_dict = {}
        if isinstance(entry, MolFileEntry):
            mol_objs_dict[entry.get_biopython_key()] = entry.mol_obj

        nb_compounds = set([x[1] for x in nb_pairs])

        perceiver = _get_perceiver(add_h, cache)
        atm_grps_mngr = perceiver.perceive_atom_groups(nb_compounds,
                                                       mol_objs_dict=mol_objs_dict)
        atm_grps_mngr.entry = entry

        return atm_grps_mngr
    
def _get_perceiver(add_h, cache=None, ph=7.4, 
                                amend_mol=True,
                                mol_obj_type='rdkit'):
    feats_factory_func = ChemicalFeatures.BuildFeatureFactory
    feature_factory = feats_factory_func(ATOM_PROP_FILE)
    feature_extractor = FeatureExtractor(feature_factory)

    perceiver = AtomGroupPerceiver(feature_extractor, add_h=add_h,
                                   ph=ph, amend_mol=amend_mol,
                                   mol_obj_type=mol_obj_type,
                                   cache=cache,
                                   tmp_path="%s/tmp" % os.getcwd())

    return perceiver