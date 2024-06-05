import multiprocessing
import os
import os.path as osp

from tqdm import tqdm

import numpy as np

import pandas as pd

import oddt
import oddt.pandas as opd
from rdkit import Chem

from torch_pgn.graphs.graph_utils import _return_graph
from torch_pgn.datasets.PGDataset import PGDataset

class ManyVsManyDataset(PGDataset):
    """
    This dataset is for a 1:1 receptor to ligand mapping. The paths (raw_pdb_path and raw_mol_path) are the same and
    point to a directory containing subdirectories containing each ligand receptor pair.
    #TODO: Currently conforms to only pdbbind input formats. Make this much more generic.
    """
    def __init__(self, args):
        super(ManyVsManyDataset, self).__init__(args)
        self.raw_data_path = args.raw_data_path
        self.num_workers = args.num_workers
        self.label_file = args.label_file
        self.label_col = args.label_col
        self.process_raw_data()
        self.write_graphs()

    def process_raw_data(self):
        raw_path = self.raw_data_path
        directories = os.listdir(raw_path)
        energy = pd.read_csv(self.label_file,
                             sep='\s+',
                             usecols=[0, self.label_col],
                             names=['name',
                                    'label'],
                             comment='#')
        energy = energy.set_index('name')
        inputs = []
        for name in tqdm(directories):
            if name not in ['index', 'readme', '.DS_Store']:
                pdb_path = os.path.join(raw_path, name, name + "_pocket.pdb")
                ligand_path = os.path.join(raw_path, name, name + "_ligand.sdf")
                try:
                    receptor = next(oddt.toolkit.readfile('pdb', pdb_path))
                    receptor.protein = True
                    ligand = opd.read_sdf(ligand_path, skip_bad_mols=True)['mol'][0]
                    if ligand is not None:
                        inputs.append((receptor, ligand, energy.loc[name, 'label'], name,
                                       self.proximity_radius, self.ligand_depth, self.receptor_depth))
                except:
                    continue

        with multiprocessing.Pool(processes=self.num_workers) as p:
            self.graphs = list(tqdm(p.imap(_return_graph, inputs)))
