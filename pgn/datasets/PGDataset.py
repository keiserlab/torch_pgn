from abc import ABC, abstractmethod

import networkx as nx
import numpy as np

from tqdm import tqdm

import os
import os.path as osp
import multiprocessing

class PGDataset(ABC):
    """Generic class for loading datasets. It has all of the methods required to write the dataset in order to be
    compatible with the rest of the processing pipeline."""
    def __init__(self, args):
        self.num_workers = args.num_workers
        self.data_path = args.data_path
        self.save_graphs = args.save_graphs
        self.graphs = []
        self.directed = args.directed

    @abstractmethod
    def process_raw_data(self):
        pass

    def write_graphs(self):
        #TODO: Fix for multiprocessing (move extra stuff to helper function and switch to imap/tqdm
        for i, entry in tqdm(enumerate(self.graphs)):
            name, graph, energy = entry
            current_dir = osp.join(self.data_path, name)
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
        current_dir = osp.join(self.data_path, prefix)
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
        return nodes, edges, node_features, edge_features, node_pos
