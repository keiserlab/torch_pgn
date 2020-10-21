import multiprocessing
import os
import os.path as osp

from tqdm import tqdm

import numpy as np

import oddt
import oddt.pandas as opd

from pgn.graphs.graph_utils import _return_graph
from pgn.datasets.PGDataset import PGDataset

class OneVsManyDataset(PGDataset):
    """
    Container class for the D4 Dopamine Receptor docking dataset.
    """
    def __init__(self, args):
        self.raw_pdb_path = args.raw_pdb_path
        self.raw_mol_path = args.raw_mol_path
        self.num_workers = args.num_workers

    def process_raw_data(self):
        receptor = next(oddt.toolkit.readfile('pdb', self.raw_pdb_path))
        data = opd.read_mol2(self.raw_mol_path)
        energy = data['Total Energy']
        name = data['Name']
        mol = data['mol']
        inputs = []
        for i, molecule in tqdm(enumerate(name), prefix="Formating raw data for graph creation"):
            inputs.append((receptor, mol[i], energy[i], molecule))

        with multiprocessing.Pool(processes=self.num_workers) as p:
            self.graphs = list(tqdm(p.imap(_return_graph, inputs)))







