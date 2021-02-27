import sys
import os
sys.path.insert(0, "/srv/home/zgaleday/pgn")
from pgn.train.run_training import run_training
from pgn.train.hyperopt import hyperopt
from pgn.args import TrainArgs, HyperoptArgs

args = HyperoptArgs()

args.from_dict({'raw_data_path': '/srv/home/zgaleday/pdbbind_general_raw',
                'label_file': '/srv/home/zgaleday/pdbbind_general_raw/index/INDEX_general_PL_data.2019',
                'data_path': '/srv/home/zgaleday/IG_data/pdbbind_general_pgn/',
                'dataset_type': 'many_v_many',
                #'split_type': 'random',
                'construct_graphs': False,
                'save_dir': '/srv/home/zgaleday/models/pgn/figure_2/general_final_pgn_protein_splits',
                'device': 'cuda:7',
                'epochs': 200,
                'cv_folds': 5,
                'save_splits': True,
                'num_iters': 20,
		'num_workers': 0,
		'batch_size': 128,
		'weight_decay': True,
                'split_type': 'defined_test',
                'split_dir': '/srv/home/zgaleday/IG_data/general_protein_splits'
                })
args.process_args()

print(args)

hyperopt(args)
