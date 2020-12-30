import sys
sys.path.append('~/pgn')
sys.path.append('~/git/pgn/')

import os.path as osp

from pgn.datasets.OneVsManyDataset import OneVsManyDataset
from pgn.datasets.ManyVsManyDataset import ManyVsManyDataset
from pgn.data.ProximityGraphDataset import ProximityGraphDataset
from pgn.data.data_utils import format_data_directory, split_test_graphs
from pgn.args import DataArgs


def process_raw(args: DataArgs):
    if args.construct_graphs:
        format_data_directory(args)
        #Add check to see if dataset already processed.
        if args.dataset_type == 'many_v_many':
            ManyVsManyDataset(args)
        elif args.dataset_type == 'one_v_many':
            OneVsManyDataset(args)
        else:
            raise ValueError("Please input a valid dataset type.")

    if args.split_type == 'defined':
        split_test_graphs(args.data_path, args.label_file)


if __name__ == "__main__":

    args = DataArgs()
    # For parsing from terminal
    # args.parse_args()
    args.from_dict({'raw_data_path': '/Users/student/git/pgn/test/working_data/toy_data/ManyVsManyToy',
                    'data_path': '/Users/student/git/pgn/test/working_data/',
                    'dataset_type': 'many_v_many',
                    'split_type': 'random',
                    'construct_graphs': True
                    })
    args.process_args()

    process_raw(args)
