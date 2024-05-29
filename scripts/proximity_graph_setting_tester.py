import sys
sys.path.insert(0, "/srv/home/zgaleday/pgn")

from pgn.args import TrainArgs, HyperoptArgs, DataArgs
from pgn.load_data import process_raw
from scripts.generate_final_correlations import generate_final_correlations

import os
import os.path as osp

def generate_datasets(raw_path_1, raw_path_2, data_path, dataset_type, mode,
                      radii=None,
                      lig_depth=None, receptor_depth=None):
    if mode == 'radius':
        radii = [1, 2.5, 3.5, 4.5, 5.5]
    if mode == 'depth':
        receptor_depth = [1, 2, 3, 4, 5, 6]
    if radii is None:
        radii = [4.5]
    if lig_depth is None:
        lig_depth = [-1]
    if receptor_depth is None:
        receptor_depth = [4]

    for radius in radii:
        for ld in lig_depth:
            for rd in receptor_depth:
                current_dir = osp.join(data_path, 'radius{0}_lig_depth_{1}_receptor_depth_{2}'.
                                       format(str(radius), str(ld), str(rd)))
                if osp.isdir(current_dir):
                    continue
                else:
                    os.mkdir(current_dir)
                    args = DataArgs()
                    if dataset_type == 'many_v_many':
                        args.from_dict({'raw_data_path': raw_path_1,
                                        'label_file': raw_path_2,
                                        'data_path': current_dir,
                                        'dataset_type': dataset_type,
                                        'split_type': 'random',
                                        'save_plots': True,
                                        'proximity_radius': radius,
                                        'ligand_depth': ld,
                                        'receptor_depth': rd,
                                        'num_workers': 64
                                        })
                    else:
                        args.from_dict({'raw_mol_path': raw_path_1,
                                        'raw_pdb_path': raw_path_2,
                                        'data_path': current_dir,
                                        'dataset_type': dataset_type,
                                        'split_type': 'random',
                                        'save_plots': False,
                                        'proximity_radius': radius,
                                        'ligand_depth': ld,
                                        'receptor_depth': rd,
                                        'num_workers': 64
                                        })

                    args.process_args()

                    process_raw(args)


def generate_repeats(data_path, checkpoint_path, save_dir, split_dir, device, epoch=250):
    for dir in os.listdir(data_path):
        try:
            current_dir = osp.join(save_dir, dir)
            os.mkdir(current_dir)
            data_dir = osp.join(data_path, dir)
            generate_final_correlations(checkpoint_path, current_dir, split_dir, device, data_path=data_dir, repeats=3, epochs=300)
        except:
            pass

if __name__ == '__main__':
    raw_data_path = sys.argv[1]
    raw_label_file = sys.argv[2]
    data_path = sys.argv[3]
    mode = sys.argv[4]
    dataset_type = 'many_v_many'
    if len(sys.argv) > 5:
        dataset_type = sys.argv[5]
    generate_datasets(raw_data_path, raw_label_file, data_path, dataset_type, mode)
    if len(sys.argv) > 6:
        checkpoint_dir = sys.argv[6]
        model_path = sys.argv[7]
        split_dir = sys.argv[8]
        device = sys.argv[9]
        generate_repeats(data_path, checkpoint_dir, model_path, split_dir, device)
