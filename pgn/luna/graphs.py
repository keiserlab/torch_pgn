# Code to handle virtual nodes if requested by the user

import networkx as nx
import numpy as np

from pgn.luna.luna_utils import calculate_interactions
from pgn.luna.featurization import featurize_virtual_nodes, luna_featurize_edge


def yield_luna_graph(mol, pdbfile, virtual = True, add_proximal = True):
    atm_grp_mngr, interactions_mngr = calculate_interactions('prot', 'mol', mol, pdbfile, 
                                                            add_proximal = add_proximal)
    
    G, node_dict = create_covalent_graph(atm_grp_mngr.graph)
    add_covalent_edges(G, atm_grp_mngr.graph, node_dict)
    
    if virtual:
        node_dict.update(make_virtual_nodes(G, interactions_mngr, node_dict, len(G.nodes)))
        
    add_luna_edges(G, interactions_mngr, node_dict, virtual = virtual)
    
    return G


def assign_node_idx(g):
    return dict([(node, i) for i, node in enumerate(g.nodes)])

def create_covalent_graph(g, virtual = False):
    node_dict = assign_node_idx(g)
    G = nx.Graph()
    
    # populate nodes and node features
    for atom, node in node_dict.items():
        molecule = 'ligand' if atom.parent.is_target() else 'receptor'
        features = np.append(atom.invariants, atom.parent.is_target())
        G.add_node(node, atom_class = molecule, features = features, pos3d = atom.atom.coord)

    return G, node_dict

def add_covalent_edges(G, g, node_dict):
    for edge in g.edges:
        #order = edge['bond_type'].value
        #aromatic = bond['aromatic']
        u, v = edge
        bond = g.edges[u, v]
        order = bond['bond_type'].value
        aromatic = bond['aromatic']

        edge_features = luna_featurize_edge(bond_type = 'covalent', order = order, aromatic = aromatic,
                                            u_pos=u.atom.coord, v_pos=v.atom.coord)
        G.add_edge(node_dict[u], node_dict[v], weight=edge_features[0], features=edge_features)
        G.add_edge(node_dict[v], node_dict[u], weight=edge_features[0], features=edge_features)
        
def make_virtual_nodes(G, im, node_dict, offset = 0):
    # get set of all atomic groups participating in interactions
    vrtl_set = set()
    vrtl_dict = {}
    for inter in im.interactions:
        for atm_grp in [inter.trgt_grp, inter.src_grp]:
            if not len(atm_grp.atoms) > 1: # only one atom, don't need to create a new node
                continue
            vrtl_set.add((frozenset(atm_grp.atoms), atm_grp.has_target()))
    
    for i, (vrtl_grp,is_ligand) in enumerate(vrtl_set):
        
        node_features, centroid = featurize_virtual_nodes(vrtl_grp)
        
        G.add_node(i + offset, atom_class = 'virtual', 
                   features = np.append(node_features, is_ligand),  pos3d = centroid)
        
        vrtl_dict[vrtl_grp] = i + offset
        for v in vrtl_grp:
            edge_features = luna_featurize_edge(bond_type='virtual', 
                                              u_pos = centroid, v_pos = v.atom.coord)
            G.add_edge(i+offset, node_dict[v], weight=edge_features[0], features=edge_features)
            G.add_edge(node_dict[v], i+offset, weight=edge_features[0],features=edge_features)

    return vrtl_dict

def add_luna_edges(G, im, node_dict, virtual = True):

    for inter in im.interactions:
        if not virtual: # all by all
            for u in inter.src_grp.atoms:
                for v in inter.trgt_grp.atoms:
                    edge_features = luna_featurize_edge(bond_type = inter.type, 
                                                        u_pos = u.atom.coord, v_pos = v.atom.coord) 
                    G.add_edge(node_dict[u], node_dict[v], weight=edge_features[0], features=edge_features)
                    G.add_edge(node_dict[v], node_dict[u], weight=edge_features[0], features=edge_features)
        else:
            # create edge between virtual nodes (or virtual + protein/ligand node)
            u = frozenset(inter.src_grp.atoms) if len(inter.src_grp.atoms) > 1 else inter.src_grp.atoms[0]
            v = frozenset(inter.trgt_grp.atoms) if len(inter.trgt_grp.atoms) > 1 else inter.trgt_grp.atoms[0]
            
            u_node, v_node = node_dict[u], node_dict[v]
            # need to be positions
            edge_features = luna_featurize_edge(bond_type=inter.type, #update
                                                u_pos = G.nodes[u_node]['pos3d'], v_pos = G.nodes[v_node]['pos3d'])
            
            G.add_edge(u_node, v_node, weight=edge_features[0],features=edge_features)
            G.add_edge(v_node, u_node,weight=edge_features[0],features=edge_features)
        