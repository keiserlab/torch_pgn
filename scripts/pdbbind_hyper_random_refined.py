import sys
import os
sys.path.insert(0, "/srv/home/zgaleday/pgn")
from pgn.train.run_training import run_training
from pgn.train.hyperopt import hyperopt
from pgn.args import TrainArgs, HyperoptArgs

args = HyperoptArgs()

args.from_dict({'raw_pdb_path': '/srv/home/zgaleday/IG_data/D4_pdbs/d4_receptor_with_h.pdb',
                'raw_mol_path': '/srv/nas/mk2/projects/D4_screen/working_data/Results/Test_Code/medium_diverse_stratified.mol2',
                'data_path': '/srv/home/zgaleday/IG_data/pdbbind_refined_2019/',
                'dataset_type': 'many_v_many',
                'split_type': 'defined_test',
                'construct_graphs': False,
                'save_dir': '/srv/home/zgaleday/models/pgn/figure_2/refined_final_pgn_protein_splits',
                'device': 'cuda:1',
                'epochs': 200,
                'cv_folds': 5,
                'save_splits': True,
                'num_iters': 20,
		'num_workers': 0,
		'batch_size': 128,
		'weight_decay': True,
                'split_dir': '/srv/home/zgaleday/IG_data/refined_protein_splits' 
                })
args.process_args()

print(args)

hyperopt(args)
