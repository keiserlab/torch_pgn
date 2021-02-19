from oddt.fingerprints import PLEC

import sys
import os
import os.path as osp
import pandas as pd
import oddt
import oddt.pandas as opd
from tqdm import tqdm

def generate_plec_pdbbind(raw_path, label_file, output_dir, dim=1024*16):
    directories = os.listdir(raw_path)
    energy = pd.read_csv(label_file,
                         sep='\s+',
                         usecols=[0, 3],
                         names=['name',
                                'label'],
                         comment='#')
    energy = energy.set_index('name')
    fps = []
    labels = []
    names = []
    print(energy)
    for name in tqdm(directories):
        if name not in ['index', 'readme', '.DS_Store']:
            pdb_path = os.path.join(raw_path, name, name + "_pocket.pdb")
            ligand_path = os.path.join(raw_path, name, name + "_ligand.sdf")
            try:
                receptor = next(oddt.toolkit.readfile('pdb', pdb_path))
                ligand = opd.read_sdf(ligand_path, skip_bad_mols=True)['mol'][0]
                if ligand is not None:
                    fp = '\t'.join(PLEC(ligand, receptor, count_bits=False).astype(str))
                    label = energy.loc[name, 'label']
                    fps.append(fp)
                    labels.append(label)
                    names.append(name)
            except:
                print('Fuck I broke')
                continue
    df = pd.DataFrame({'name': names, 'fp': fps, 'label': labels})
    df.to_csv(osp.join(output_dir, 'formatted_plec.csv'), index=False)


def generate_plec_d4(raw_pdb_path, raw_mol_path, output_dir, dim=1024*16):
    receptor = next(oddt.toolkit.readfile('pdb', raw_pdb_path))
    data = opd.read_mol2(raw_mol_path)
    energy = data['Total Energy']
    names = data['Name']
    mol = data['mol']
    fps = []
    labels = []
    names = []
    for i, name in enumerate(names):
        ligand = mol[i]
        label = energy[i]
        if ligand is not None:
            fp = '\t'.join(PLEC(ligand, receptor, count_bits=False).astype(str))
            label = label
            fps.append(fp)
            labels.append(label)
            names.append(name)
    df = pd.DataFrame({'name': names, 'fp': fps, 'label': labels})
    df.to_csv(osp.join(output_dir, 'formatted_plec.csv'), index=False)


if __name__ == "__main__":
    type = sys.argv[1]
    path_1 = sys.argv[2]
    path_2 = sys.argv[3]
    output_path = sys.argv[4]
    if type == 'pdbbind':
        generate_plec_pdbbind(path_1,
                              path_2,
                              output_path)
    elif type == 'd4':
        generate_plec_d4(path_1,
                         path_2,
                         output_path)
