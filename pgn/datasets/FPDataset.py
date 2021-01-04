import pandas as pd
import numpy as np

import os.path as osp

NAME_COLUMN = 'name'
FP_DELIM = '\t'

class FPDataset():
    """
    Dataset class to load fingerprint data.
    """
    def __init__(self, args):
        self.num_workers = args.num_workers
        self.data_path = args.data_path
        self.fp_format = args.fp_format
        self.fp_column = args.fp_column
        self.fp_dim = args.fp_dim
        self.label_column = args.label_col
        self.raw_data_path = args.raw_data_path
        self.process_raw_data()
        self.write_fps()

    def process_raw_data(self):
        """Load the raw data and format to be compatible with FingerprintDataset"""
        converter = {self.fp_column: lambda fp: np.array(fp.split(FP_DELIM)).astype(int)}
        df = pd.read_csv(self.raw_data_path, converters=converter)
        self.label_col_name = df.columns[self.label_column]
        df = df[[NAME_COLUMN, self.fp_column, self.label_col_name]]
        fps = df[self.fp_column].values
        if self.fp_format == 'sparse':
            fps = [self._to_dense(fp) for fp in fps]
        df[self.fp_column] = fps
        self.df = df

    def write_fps(self):
        names = np.array(self.df[NAME_COLUMN].values).astype(str)
        fps = np.vstack(self.df[self.fp_column].values)
        labels = self.df[self.label_col_name].values
        base_dir = osp.join(self.data_path, 'raw', 'train')
        np.save(osp.join(base_dir, 'names.npy'), names)
        np.save(osp.join(base_dir, 'labels.npy'), labels)
        np.save(osp.join(base_dir, 'fps.npy'), fps)

    def _to_dense(self, fp):
        dense_fp = np.zeros(self.fp_dim)
        dense_fp[list(fp)] += 1
        return dense_fp

