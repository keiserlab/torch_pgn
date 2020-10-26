from pgn.datasets.OneVsManyDataset import OneVsManyDataset
from pgn.datasets.ManyVsManyDataset import ManyVsManyDataset
from pgn.data.ProximityGraphDataset import ProximityGraphDataset
from pgn.data.data_utils import format_data_directory
from pgn.args import DataArgs


args = DataArgs()
# For parsing from terminal
#args.parse_args()
args.from_dict({'raw_data_path': '/Users/student/git/pgn/toy_data/ManyVsManyToy',
                'data_path': '/Users/student/git/pgn/toy_out',
                'dataset_type': 'many_v_many',
                'split_type': 'random',
                })
args.process_args()

if args.construct_graphs:
    format_data_directory(args)
    #Add check to see if dataset already processed.
    if args.dataset_type == 'many_v_many':
        ManyVsManyDataset(args)
    elif args.dataset_type == 'one_v_many':
        OneVsManyDataset(args)
    else:
        raise ValueError("Please input a valid dataset type.")

dataset = ProximityGraphDataset(args.dataset_type)