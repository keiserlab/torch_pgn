from luna.util.default_values import INTERACTION_IDS
import numpy as np

def featurize_virtual_nodes(atoms):
    # calculate centroid
    centroid = np.mean(np.stack([a.atom.coord for a in atoms]), axis = 0)
    
    # since we're using the LUNA atomic invariants, the order is differeit - 3rd val is atomic #
    invariants = np.mean(np.stack([a.invariants for a in atoms]), axis = 0)
    invariants[2] = 0
    return invariants, centroid

def luna_featurize_edge(u_pos = None, v_pos = None, 
                        bond_type='covalent', order = -1, aromatic = False):
    molecule, virtual, typed = False, False, False
    luna_edge_feats = np.zeros(44) #44 possible interaction types, see LUNA utils for definition
    
    if bond_type == 'covalent':
        molecule = True
    elif bond_type == 'virtual':
        virtual = True
    else: # interaction edge
        luna_edge_feats[INTERACTION_IDS[bond_type]] = 1

    return np.concatenate((np.array([
              np.sum(np.sqrt(np.square((u_pos - v_pos)))), # euclidean distance,
              int(order == 1), int(order == 2), int(order == 3),
              int(aromatic), int(molecule), int(virtual) 
        ]), luna_edge_feats))