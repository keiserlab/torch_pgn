from abc import ABC, abstractmethod

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import os
import os.path as osp
import multiprocessing

class PGDataset(ABC):
    """Generic class for loading datasets. It has all of the methods required to write the dataset in order to be
    compatible with the rest of the processing pipeline."""
    def __init__(self, args):
        self.args = args
        self.num_workers = args.num_workers
        self.data_path = args.data_path
        self.save_graphs = args.save_graphs
        self.graphs = []
        self.directed = args.directed
        self.save_plots = args.save_plots
        self.proximity_radius = args.proximity_radius
        self.ligand_depth = args.ligand_depth
        self.receptor_depth = args.receptor_depth

    @abstractmethod
    def process_raw_data(self):
        pass

    def write_graphs(self):
        #TODO: Fix for multiprocessing (move extra stuff to helper function and switch to imap/tqdm
        if not osp.isdir(self.data_path):
            os.mkdir(self.data_path)
        for i, entry in tqdm(enumerate(self.graphs)):
            name, graph, energy = entry
            current_dir = osp.join(self.data_path, 'raw', 'train', name)
            if not os.path.isdir(current_dir):
                os.mkdir(current_dir)
                if i % 100 == 0:
                    print("Writing " + name)
                node_num = len(graph.nodes)
                edge_num = len(graph.edges)
                label_path = osp.join(current_dir, name + "_label")
                label = np.array(([energy]))
                np.save(label_path, label)
                self._write_graph(graph, prefix=name)

    def _write_graph(self, G, prefix=''):
        """
        Takes a networkx graph object and writes to files.
        x: A csv of node features of the shape [num_nodes, num_node_features]
        edge_index: An adjacency representation of the graph of shape [2, number of edges] format is:
            [u1, u2, u3, ...., un]
            [v1, v2, v3, ..., vn]
            where ui is source and vi is sink.
        edge_features: A csv of edge feature of the shape [num_edges, num_edge_features]
        :param G: A networkx graph object.
        :param prefix: The prefix to add to the default filenames.
        :return: None
        """
        current_dir = osp.join(self.data_path, 'raw', 'train', prefix)
        # if osp.isdir(current_dir):
        #     count = 1
        #     while osp.isdir(current_dir):
        #         current_dir = osp.join(self.data_path, 'raw', 'train', prefix + '_pose{0}'.format(count))
        #         count += 1
        prefix = prefix + '_'
        nodes = np.array(G.nodes)
        edges = G.edges
        node_feature_dict = nx.get_node_attributes(G, 'features')
        edge_feature_dict = nx.get_edge_attributes(G, 'features')
        node_pos_dict = nx.get_node_attributes(G, 'pos3d')
        node_features = np.array([node_feature_dict[n] for n in nodes])
        node_pos = np.array([node_pos_dict[n] for n in nodes])
        edge_features = np.array([edge_feature_dict[e] for e in edges])
        edges = np.array(G.edges).T
        if not self.directed:
            edges = np.hstack((edges, edges[::-1, :]))
            edge_features = np.vstack((edge_features, edge_features))
        np.save(osp.join(current_dir, prefix + 'node_features'), node_features)
        np.save(osp.join(current_dir, prefix + 'edges'), edges)
        np.save(osp.join(current_dir, prefix + 'edge_features'), edge_features)
        np.save(osp.join(current_dir, prefix + 'pos3d'), node_pos)
        if self.save_plots:
            self._visualize_graph(G, node_pos, node_features, current_dir)
        self._output_interacting_set(G, current_dir)
        return nodes, edges, node_features, edge_features, node_pos

    def _visualize_graph(self, G, node_pos, node_feats, save_dir):
        #Assumes ligand toggle is last position feature matrix and 1 for ligand 0 for protein
        ligand_nodes = np.where(node_feats[:, -1] == 1)[0]
        prot_nodes = np.where(node_feats[:, -1] == 0)[0]
        nx.draw_networkx_edges(G, node_pos[:,:-1], alpha=0.6)
        nx.draw_networkx_nodes(G, node_pos[:,:-1],
                               nodelist=list(ligand_nodes),
                               node_color='r',
                               alpha=0.6)
        if prot_nodes.size > 0:
            nx.draw_networkx_nodes(G, node_pos[:, :-1],
                                   nodelist=list(prot_nodes),
                                   node_color='b',
                                   alpha=0.6)
        plt.axis('off')
        plt.title(G.name)
        plt.savefig(osp.join(save_dir, G.name + '_prox_graph_proj.png'))
        plt.clf()

    def _output_interacting_set(self, G, save_dir):
        pdb_idx = np.array(list(nx.get_node_attributes(G, 'pdb_idx').values()))
        np.save(osp.join(save_dir, G.name + "_interacting_idx"), pdb_idx)