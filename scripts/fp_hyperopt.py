import sys
import os
sys.path.insert(0, "/srv/home/zgaleday/pgn")
from pgn.train.run_training import run_training
from pgn.train.hyperopt import hyperopt
from pgn.args import TrainArgs, HyperoptArgs

args = HyperoptArgs()

args.from_dict({'raw_data_path': '/srv/home/zgaleday/IG_data/pdbbind_refined_16384/formatted_plec.csv',
                'data_path': '/srv/home/zgaleday/IG_data/pgn_pdbbind_refined_plec_16384/',
                'search_keys': ['ffn_hidden_size', 'ffn_num_layers', 'dropout'],
                'dataset_type': 'fp',
                'encoder_type': 'fp',
                'fp_dim': 1024*16,
                'split_type': 'random',
                'construct_graphs': True,
                'save_dir': '/srv/home/zgaleday/models/pgn/figure_2/pdbbind_refined_rand_PLEC_hyper',
                'device': 'cuda:5',
                'epochs': 30,
                'cv_folds': 5,
                'save_splits': True,
                'num_iters': 15,
                'num_workers': 0,
                'label_col': 2,
                'batch_size': 128,
                'weight_decay': True})
args.process_args()

print(args)

hyperopt(args)
