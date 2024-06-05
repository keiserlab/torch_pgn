import sys
import os
sys.path.insert(0, "/srv/home/zgaleday/torch_pgn")
from torch_pgn.train.run_training import run_training
from torch_pgn.train.hyperopt import hyperopt
from torch_pgn.args import TrainArgs, HyperoptArgs

args = HyperoptArgs()

args.from_dict({'raw_pdb_path': 'vi',
                'raw_mol_path': '/srv/nas/mk2/projects/D4_screen/working_data/Results/Test_Code/medium_diverse_stratified.mol2',
                'data_path': '/srv/home/zgaleday/IG_data/D4_graphs_dist',
                'dataset_type': 'one_v_many',
                'split_type': 'random',
                'construct_graphs': False,
                'save_dir': '/srv/home/zgaleday/models/torch_pgn/figure_2/d4_final_pgn',
                'device': 'cuda:6',
                'epochs': 150,
                'cv_folds': 5,
                'save_splits': True,
                'num_iters': 20,
                'batch_size': 128,
                'num_workers': 0,
                'multi_gpu': False,
                'weight_decay': True})
args.process_args()

print(args)

hyperopt(args)
