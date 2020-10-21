import multiprocessing
import os
import os.path as osp

from tqdm import tqdm

import numpy as np

import oddt
import oddt.pandas as opd

from pgn.graphs.graph_utils import _return_graph
from pgn.datasets.PGDataset import PGDataset

class ManyVsManyDataset(PGDataset):
    """
    This dataset is for a 1:1 receptor to ligand mapping. The paths (raw_pdb_path and raw_mol_path) are the same and
    point to a directory containing subdirectories containing each ligand receptor pair.
    #TODO: Currently conforms to only pdbbind input formats. Make this much more generic.
    """
    def __init__(self, args):
        super(ManyVsManyDataset, self).__init__(args)
        self.raw_pdb_path = args.raw_pdb_path
        self.raw_mol_path = args.raw_mol_path
        self.num_workers = args.num_workers
        self.label_file = args.label_file
        self.label_col_name = args.label_col_name

    def process_raw_data(self):
        raw_path = self.raw_pdb_path
        directories = os.listdir(self.raw_pdb_path)

        inputs = []

        for name in tqdm(directories):
            if name not in ['index', 'readme', '.DS_Store']:
                pdb_path = os.path.join(raw_path, name, name + "_protein.pdb")
                ligand_path = os.path.join(raw_path, name, name + "_ligand.sdf")
                receptor = next(oddt.toolkit.readfile('pdb', pdb_path))
                ligand = opd.read_sdf(ligand_path)['mol'][0]
                inputs.append((receptor, ligand, energy[i], name))

        with multiprocessing.Pool(processes=self.num_workers) as p:
            self.graphs = list(tqdm(p.imap(_return_graph, inputs)))