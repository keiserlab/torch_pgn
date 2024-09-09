import sys
sys.path.insert(0, "/srv/home/zgaleday/torch_pgn")

from torch_pgn.datasets.OneVsManyDataset import OneVsManyDataset
from torch_pgn.datasets.ManyVsManyDataset import ManyVsManyDataset
from torch_pgn.datasets.FPDataset import FPDataset
from torch_pgn.data.data_utils import format_data_directory
from torch_pgn.args import DataArgs


def process_raw(args: DataArgs):
    if args.construct_graphs:
        format_data_directory(args)
        #Add check to see if dataset already processed.
        if args.dataset_type == 'many_v_many':
            ManyVsManyDataset(args)
        elif args.dataset_type == 'one_v_many':
            OneVsManyDataset(args)
        elif args.dataset_type == 'fp':
            FPDataset(args)
        else:
            raise ValueError("Please input a valid dataset type.")


def process_raw_data():
    """
    Parses raw data for dataset generation independent of training. Useful for testing data generation for new
    datasets.

    This is an entry point for the command line code torch_pgn_process_raw
    """
    args = DataArgs()
    # For parsing from terminal
    args.parse_args()
    args.process_args()
    process_raw(args)
