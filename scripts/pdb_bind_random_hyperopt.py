import sys
import os
sys.path.insert(0, "/srv/home/zgaleday/pgn")
from pgn.train.run_training import run_training
from pgn.train.hyperopt import hyperopt
from pgn.args import TrainArgs, HyperoptArgs
import warnings

warnings.filterwarnings('ignore')
args = HyperoptArgs()

search_keys = ['num_blocks', 'int_emb_size', 'nn_conv_internal_dim',
        'basis_emb_size', 'out_emb_channels']

args.from_dict({'raw_data_path': '/srv/home/zgaleday/pdbbind_general_raw',
                'label_file': '/srv/home/zgaleday/pdbbind_general_raw/index/INDEX_general_PL_data.2019',
                'data_path': '/srv/home/zgaleday/IG_data/pyg_update_ds/pdbbind_refined_2019/',
                'dataset_type': 'many_v_many',
                'encoder_type': 'dimenet++',
                #'split_type': 'random',
                'construct_graphs': False,
                'save_dir': '/srv/home/zgaleday/models/pgn/review_experiments/final_dimenet_hyperopt',
                'device': 'cuda:4',
                'epochs': 1,
                'cv_folds': 3,
                'save_splits': True,
                'num_iters': 15,
		        'num_workers': 0,
		        'batch_size': 128,
		        'weight_decay': False,
                'split_type': 'defined_test',
                'split_dir': '/srv/home/zgaleday/models/pgn/figure_2/refined_final_pgn/splits',
                'search_keys': search_keys,
		'seed': 25
                })
args.process_args()

print(args)

hyperopt(args)
