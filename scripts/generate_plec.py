from oddt.fingerprints import PLEC
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
                continue
    df = pd.DataFrame({'name': names, 'fp': fps, 'label': labels})
    df.to_csv(osp.join(output_dir, 'formatted_plec.csv'), index=False)

if __name__ == "__main__":
    generate_plec_pdbbind('/srv/home/zgaleday/pdbbind_general_raw',
                          '/srv/home/zgaleday/pdbbind_general_raw/index/INDEX_general_PL_data.2019',
                          '/srv/home/zgaleday/IG_data/pdbbind_general_16384')