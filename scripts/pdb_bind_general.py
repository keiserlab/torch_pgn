import sys
import os

import openbabel
sys.path.insert(0, "/srv/home/zgaleday/pgn")
from pgn.train.run_training import run_training
from pgn.args import TrainArgs, HyperoptArgs

import torch

args = TrainArgs()

args.from_dict({'raw_data_path': '/srv/home/zgaleday/pdbbind_general_raw',
                'data_path': '/srv/home/zgaleday/IG_data/pdbbind_general_pgn',
                'label_file': '/srv/home/zgaleday/pdbbind_general_raw/index/INDEX_general_PL_data.2019',
                'dataset_type': 'many_v_many',
                'split_type': 'random',
                'construct_graphs': False,
                'save_dir': '/srv/home/zgaleday/models/pgn/pdbbind_general_random_300',
                'validation_percent': 0.05,
                'test_percent': 0.05,
                'device': 'cuda:5',
                'num_workers': 0,
                'epochs': 300,
                'num_layers': 3,
                'ff_dim_1': 1024,
                'depth': 3,
                'weight_decay': True,
                'hidden_dim': 200
                })
args.process_args()

print(args)

trainer = run_training(args)
