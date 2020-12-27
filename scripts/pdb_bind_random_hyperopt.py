import sys
import os
sys.path.insert(0, "/srv/home/zgaleday/pgn")
from pgn.train.run_training import run_training
from pgn.train.hyperopt import hyperopt
from pgn.args import TrainArgs, HyperoptArgs

args = HyperoptArgs()

args.from_dict({'raw_pdb_path': '/srv/home/zgaleday/IG_data/D4_pdbs/d4_receptor_with_h.pdb',
                'raw_mol_path': '/srv/nas/mk2/projects/D4_screen/data/Results/Test_Code/medium_diverse_stratified.mol2',
                'data_path': '/srv/home/zgaleday/IG_data/pdb_bind_random/',
                'dataset_type': 'one_v_many',
                'split_type': 'random',
                'construct_graphs': False,
                'save_dir': '/srv/home/zgaleday/models/pgn/figure_2/pdbbind_pgn_rand_hyper',
                'device': 'cuda:0',
                'epochs': 150,
                'cv_folds': 5,
                'save_splits': True,
                'num_iters': 20,
		        'batch_size': 128
                })
args.process_args()

print(args)

hyperopt(args)