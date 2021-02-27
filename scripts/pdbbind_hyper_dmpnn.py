import sys
import os
sys.path.insert(0, "/srv/home/zgaleday/pgn")
from pgn.train.run_training import run_training
from pgn.train.hyperopt import hyperopt
from pgn.args import TrainArgs, HyperoptArgs

args = HyperoptArgs()

args.from_dict({'raw_pdb_path': '/srv/home/zgaleday/IG_data/D4_pdbs/d4_receptor_with_h.pdb',
                'raw_mol_path': '/srv/nas/mk2/projects/D4_screen/working_data/Results/Test_Code/medium_diverse_stratified.mol2',
                'data_path': '/srv/home/zgaleday/IG_data/d4_graphs_pgn/',
                'dataset_type': 'one_v_many',
                'encoder_type': 'dmpnn',
                #'split_type': 'defined_test',
                #'split_dir': '/srv/home/zgaleday/IG_data/general_protein_splits',
                'construct_graphs': False,
                'save_dir': '/srv/home/zgaleday/models/pgn/figure_2/d4_final_dmpnn/',
                'device': 'cuda:1',
                'epochs': 100,
                'cv_folds': 3,
                'save_splits': True,
                'num_iters': 20,
		'num_workers': 0,
		'batch_size': 128,
                'multi_gpu': False,
		'weight_decay': True,
                'search_keys': ['depth', 'ffn_num_layers', 'dropout', 'ffn_hidden_size'] 
                })
args.process_args()

print(args)

hyperopt(args)
