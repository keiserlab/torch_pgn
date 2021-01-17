import sys
import os
sys.path.insert(0, "/srv/home/zgaleday/pgn")

import os.path as osp
from scripts.generate_final_correlations import generate_final_correlations

base_dir = '/srv/home/zgaleday/models/pgn/figure_2'
model_dirs = ['pdbbind_general_rand_fp_hyper/',
               'refined_final_pgn/',
               'pdbbind_general_rand_hyper_fixcv/']
checkpoint_dirs = ['dropout_0p15000000000000002_ffn_hidden_size_1900_ffn_num_layers_5/cv_fold_0',
                   'dropout_0p05_ffn_hidden_size_600_ffn_num_layers_4_fp_dim_1024/cv_fold_0',
                   'dropout_0p35000000000000003_ffn_hidden_size_2200_ffn_num_layers_2_fp_dim_4096/cv_fold_0']


final_dirs = ['PLEC_pdbbind_general_random', 'PGN_pdbbind_refined', 'PGN_pdbbind_general']
epochs = [100, 250, 400]

for i in range(len(checkpoint_dirs)):
    model_dir = model_dirs[i]
    check_dir = checkpoint_dirs[i]
    final_dir = final_dirs[i]
    checkpoint_path = osp.join(base_dir, model_dir, check_dir, 'best_checkpoint.pt')
    split_path = osp.join(base_dir, model_dir, 'splits')
    final_path = osp.join(base_dir, 'repeats', final_dir)
    device = 'cuda:3'
    try:
        generate_final_correlations(checkpoint_path, final_path, split_path, device, epochs=epochs[i])
    except:
        pass
