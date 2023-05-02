from pgn.train.run_training import run_training
from pgn.train.Trainer import Trainer
from pgn.train.hyperopt import hyperopt
from pgn.args import TrainArgs, HyperoptArgs, DataArgs
from pgn.load_data import process_raw
from scripts.generate_plec import generate_plec_pdbbind
from scripts.generate_final_correlations import generate_final_correlations

import os
import os.path as osp
import numpy as np
import networkx as nx

def test_many_v_many_dataloading():

    args = DataArgs()

    args.from_dict({'raw_data_path': '/Users/student/git/pgn/tests/working_data/ManyVsManyToy',
                    'data_path': '/Users/student/git/pgn/tests/output_data/mvm_toy_dataset',
                    'dataset_type': 'many_v_many',
                    'split_type': 'random'
                    })

    args.process_args()

    process_raw(args)


def test_one_v_many_dataloading():

    args = DataArgs()

    args.from_dict({'raw_mol_path': '/Users/student/git/pgn/tests/working_data/OneVsManyToy/medium_diverse_toy.mol2',
                    'raw_pdb_path': '/Users/student/git/pgn/tests/working_data/OneVsManyToy/d4_receptor.pdb',
                    'data_path': '/Users/student/git/pgn/tests/output_data/ovm_toy_dataset',
                    'dataset_type': 'one_v_many'
                    })

    args.process_args()

    process_raw(args)

def test_pfp_train():
    args = TrainArgs()

    args.from_dict({'raw_data_path': '/Users/student/git/pgn/tests/working_data/ManyVsManyToy',
                    'data_path': '/Users/student/git/pgn/tests/working_data/toy_out',
                    'dataset_type': 'many_v_many',
                    'split_type': 'random',
                    'construct_graphs': False,
                    'save_dir': '/Users/student/git/pgn/tests/output/pfp',
                    'epochs': 5,
                    'validation_percent': 0.2,
                    'test_percent': 0.2,
                    'save_splits': True
                    })
    args.process_args()

    run_training(args)


def test_hyperopt():
    args = HyperoptArgs()

    args.from_dict({'raw_data_path': '/Users/student/git/pgn/tests/working_data/ManyVsManyToy',
                    'data_path': '/Users/student/git/pgn/tests/working_data/toy_out',
                    'dataset_type': 'many_v_many',
                    'split_type': 'random',
                    'construct_graphs': False,
                    'save_dir': '/Users/student/git/pgn/tests/hyperopt_output',
                    'epochs': 2,
                    'save_splits': True,
                    'cv_folds': 2,
                    'num_iters': 2
                    })
    args.process_args()

    hyperopt(args)


def test_dmpnn_train():

    args = TrainArgs()

    args.from_dict({'raw_data_path': '/Users/student/git/pgn/tests/working_data/ManyVsManyToy',
                    'data_path': '/Users/student/git/pgn/tests/working_data/toy_out',
                    'dataset_type': 'many_v_many',
                    'split_type': 'random',
                    'construct_graphs': False,
                    'save_dir': '/Users/student/git/pgn/tests/output/pfp',
                    'epochs': 5,
                    'validation_percent': 0.25,
                    'test_percent': 0.25,
                    'save_splits': True,
                    'split_dir': '/Users/student/git/pgn/tests/splits',
                    'encoder_type': 'dmpnn'
                    })
    args.process_args()

    run_training(args)


def test_ggnet_train():

    args = TrainArgs()

    args.from_dict({'raw_data_path': '/Users/student/git/pgn/tests/working_data/ManyVsManyToy',
                    'data_path': '/Users/student/git/pgn/tests/working_data/toy_out',
                    'dataset_type': 'many_v_many',
                    'split_type': 'random',
                    'construct_graphs': False,
                    'save_dir': '/Users/student/git/pgn/tests/output',
                    'epochs': 5,
                    'validation_percent': 0.2,
                    'test_percent': 0.2,
                    'save_splits': True,
                    'split_dir': '/Users/student/git/pgn/tests/splits',
                    'encoder_type': 'ggnet'
                    })
    args.process_args()

    run_training(args)


def test_fp_dataloading():

    args = DataArgs()

    args.from_dict({'raw_data_path': '/Users/student/git/pgn/tests/working_data/FPToy/formated_plec_toy.csv',
                    'data_path': '/Users/student/git/pgn/tests/working_data/fp_out',
                    'split_type': 'random',
                    'dataset_type': 'fp',
                    'fp_format': 'sparse',
                    'fp_dim': 1024 * 16,
                    'label_col': 2
                    })

    args.process_args()

    process_raw(args)


def test_fp_train():

    args = TrainArgs()

    args.from_dict({'raw_data_path': '/Users/student/git/pgn/tests/working_data/FPToy/formated_plec_toy.csv',
                    'data_path': '/Users/student/git/pgn/tests/working_data/fp_out',
                    'construct_graphs': False,
                    'split_type': 'random',
                    'dataset_type': 'fp',
                    'encoder_type': 'fp',
                    'fp_format': 'sparse',
                    'fp_dim': 1024 * 16,
                    'label_col': 2,
                    'save_dir': '/Users/student/git/pgn/tests/output',
                    'epochs': 5,
                    'validation_percent': 0.2,
                    'test_percent': 0.2,
                    'save_splits': True})

    args.process_args()

    run_training(args)


def test_generate_plec_pdbbind():

    generate_plec_pdbbind('working_data/ManyVsManyToy', 'working_data/ManyVsManyToy/index/INDEX_general_PL_data.2019')


def test_generate_final_correlations():
    generate_final_correlations('/Users/student/git/pgn/tests/output/pfp/cv_fold_0/best_checkpoint.pt',
                                '/Users/student/git/pgn/tests/output/final_corr',
                                split_path='/Users/student/git/pgn/tests/output/pfp/splits',
                                device='cpu',
                                epochs=5)


def test_ligand_only_dataset():
    args = TrainArgs()

    args.from_dict({'raw_data_path': '/Users/student/git/pgn/tests/working_data/ManyVsManyToy',
                    'data_path': '/Users/student/git/pgn/tests/working_data/toy_out',
                    'dataset_type': 'many_v_many',
                    'ligand_only': True,
                    'split_type': 'random',
                    'construct_graphs': False,
                    'save_dir': '/Users/student/git/pgn/tests/output/pfp',
                    'epochs': 5,
                    'validation_percent': 0.2,
                    'test_percent': 0.2,
                    'save_splits': True
                    })

    args.process_args()

    trainer = Trainer(args)
    trainer.load_data()


def test_no_interactions_dataset():
    args = TrainArgs()

    args.from_dict({'raw_data_path': '/Users/student/git/pgn/tests/working_data/ManyVsManyToy',
                    'data_path': '/Users/student/git/pgn/tests/working_data/toy_out',
                    'dataset_type': 'many_v_many',
                    'interaction_edges_removed': True,
                    'split_type': 'random',
                    'construct_graphs': False,
                    'save_dir': '/Users/student/git/pgn/tests/output/pfp',
                    'epochs': 5,
                    'validation_percent': 0.2,
                    'test_percent': 0.2,
                    'save_splits': True
                    })

    args.process_args()

    trainer = Trainer(args)
    trainer.load_data()

def test_ligand_only_readout():

    args = TrainArgs()

    args.from_dict({'raw_data_path': '/Users/student/git/pgn/tests/working_data/ManyVsManyToy',
                    'data_path': '/Users/student/git/pgn/tests/working_data/toy_out',
                    'dataset_type': 'many_v_many',
                    'split_type': 'random',
                    'encoder_type': 'ggnet',
                    'construct_graphs': False,
                    'save_dir': '/Users/student/git/pgn/tests/output/pfp',
                    'epochs': 5,
                    'validation_percent': 0.2,
                    'test_percent': 0.2,
                    'save_splits': True,
                    'ligand_only_readout': True
                    })
    args.process_args()

    run_training(args)


def test_split_conv():

    args = TrainArgs()

    args.from_dict({'raw_data_path': '/Users/student/git/pgn/tests/working_data/ManyVsManyToy',
                    'data_path': '/Users/student/git/pgn/tests/working_data/toy_out',
                    'dataset_type': 'many_v_many',
                    'split_type': 'random',
                    'encoder_type': 'pfp',
                    'construct_graphs': False,
                    'save_dir': '/Users/student/git/pgn/tests/output/pfp',
                    'epochs': 5,
                    'validation_percent': 0.2,
                    'test_percent': 0.2,
                    'save_splits': True,
                    'one_step_convolution': False
                    })
    args.process_args()

    run_training(args)

#TODO: Split into seperate helper methods file
def _load_graph(path, name):
    """
    Load graph from raw numpy files for comparision in the compare graph method
    :param path: Path to the directory containing the raw graph saved as np files
    :return: graph as a networkx object
    """
    edge_features = np.load(osp.join(path, f'{name}_edge_features.npy'))
    edges = np.load(osp.join(path, f'{name}_edges.npy'))
    label = np.load(osp.join(path, f'{name}_label.npy'))
    node_features = np.load(osp.join(path, f'{name}_node_features.npy'))
    node_pos = np.load(osp.join(path, f'{name}_pos3d.npy'))
    G = nx.Graph()
    for i in range(len(node_features)):
        features = node_features[i]
        pos = node_pos[i]
        G.add_node(i, features=features, pos3d=pos)
    for i in range(edges.shape[1]):
        u, v = edges[0, i], edges[1, i]
        features = edge_features[i]
        G.add_edge(u, v, features=features)
    G.graph['label'] = label
    return G

def _reorder_graph(G):
    """
    Function that takes a proximity graph and relabels the nodes such that it is numbered from smallest pos3d attribute
    to largest.
    :param G: Proximity Graph object
    :return: Copy of the graph with the nodes sorted as described
    """
    nodes_pos_dict = nx.get_node_attributes(G, 'pos3d')
    node_nums = sorted(nodes_pos_dict.keys())
    node_pos_list = np.array([nodes_pos_dict[k] for k in node_nums])
    sort_ind = np.lexsort((node_pos_list[:, 2], node_pos_list[:, 1], node_pos_list[:, 0]))
    node_map = {sort_ind[i]: i for i in range(len(node_nums))}
    H = nx.relabel_nodes(G, node_map)
    return H


def _compare_graphs(graph1, graph2, epsilon=0.001):
    """
    Compare proximity graphs and returns boolean for identity as defined as isomorphism and attribute matching.
    :param graph1: comparison graph 1 (networkx)
    :param G1: comparison graph 2 (networkx)
    :return:
    """
    sorted_g1 = _reorder_graph(graph1)
    sorted_g2 = _reorder_graph(graph2)
    # Compare graph size
    assert len(sorted_g1.nodes()) == len(sorted_g2.nodes())
    #compare 3d coordinate
    g1_pos = nx.get_node_attributes(sorted_g1, 'pos3d')
    g2_pos = nx.get_node_attributes(sorted_g2, 'pos3d')
    distance_matrix = [np.sqrt(np.sum(np.square(g1_pos[i] - g2_pos[i]))) for i in g1_pos.keys()]
    # TODO: Switch from global epsilon to local epsilon with node level reporting
    assert np.sum(distance_matrix) < epsilon
    #compare other node features
    g1_features = nx.get_node_attributes(sorted_g1, 'features')
    g2_features = nx.get_node_attributes(sorted_g2, 'features')
    feature_comparison = [np.array_equal(g1_features[i], g2_features[i]) for i in g1_features.keys()]
    assert np.all(feature_comparison)
    #compare adj matrix
    g1_edges = sorted(sorted_g1.edges())
    g2_edges = sorted(sorted_g2.edges())
    assert np.array_equal(g1_edges, g2_edges)
    #compare edge distances within epsilon
    g1_efs = nx.get_edge_attributes(sorted_g1, 'features')
    g2_efs = nx.get_edge_attributes(sorted_g2, 'features')
    g1_dists = np.array([g1_efs[e][0] for e in g1_edges])
    g2_dists = np.array([g2_efs[e][0] for e in g2_edges])
    dists_delta = np.abs(g1_dists - g2_dists)
    assert np.sum(dists_delta) < epsilon
    #compare categorical edge features
    g1_cat_feats = np.array([g1_efs[e][1:] for e in g1_edges])
    g2_cat_feats = np.array([g2_efs[e][1:] for e in g2_edges])
    assert (np.array_equal(g1_cat_feats, g2_cat_feats))




def _compare_graph_dataset(truth_dataset, test_dataset):
    """
    Compare two raw datasets loaded using either the OneVsManyDataset or ManyVsManyDataset classes. Requires a
    ground truth dataset and a dataset that you want to test for functional equiv. General usage is expect to be for
    changes made to graph loading. i.e. make change and ensure no difference in graph results.
    :param truth_dataset: path to top level of ground truth dataset generated before changes implemented
    :param test_dataset: path to top level of dataset generated after changed to dataset generation have been implemented
    :return: Boolean of whether datasets are functionally equivalent
    """
    #Test whether all molecule/receptor graphs are generated
    #TODO: change to testing from assert
    gt_graphs = os.listdir(osp.join(truth_dataset, 'raw', 'train'))
    test_graphs = os.listdir(osp.join(test_dataset, 'raw', 'train'))
    assert len(gt_graphs) == len(test_graphs), "Number of graphs not equal in new and ground truth dataset"
    for tg in test_graphs:
        assert tg in gt_graphs, f'{tg} not found in ground truth graphs'
        test_graph = _load_graph(osp.join(osp.join(test_dataset, 'raw', 'train', tg)), tg)
        gt_graph = _load_graph(osp.join(osp.join(truth_dataset, 'raw', 'train', tg)), tg)
        print(_compare_graphs(test_graph, gt_graph))
        break


_compare_graph_dataset('ground_truth_dataset/mvm_toy_out', 'output_data/mvm_toy_dataset')
