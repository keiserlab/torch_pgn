import pandas as pd
import numpy as np
import gzip
import os.path as osp
import os
import shutil
import oddt.pandas as opd
from tqdm import tqdm

TEST_FILE = '/srv/home/zgaleday/IG_data/d4_test_compounds/experimally_test_chunkmap.csv'
SCREEN_DIRECTORY = '/srv/nas/mk2/projects/D4_screen/'
TEST_CHUNK = 'vs_run1_chunk28150'
OUTDIR = '/srv/home/zgaleday/IG_data/d4_test_compounds/'

def parse_chunk(chunk_id, mol_name):
    temp_file = osp.join(OUTDIR, 'temp.mol2')
    with gzip.open(osp.join(SCREEN_DIRECTORY, chunk_id, 'test.mol2.gz'), 'rb') as f:
        with open(temp_file, 'wb') as f_out:
            shutil.copyfileobj(f, f_out)
    chunk = opd.read_mol2(temp_file)
    chunk['mol_name'] = chunk['mol_name'].apply(lambda x: x.split()[0])
    os.remove(temp_file)
    return chunk[chunk['mol_name'] == mol_name]

exp_df = pd.read_csv(TEST_FILE)

mol2_list = []
for zid, chunk in tqdm(zip(exp_df['ZINC ID'].values, exp_df['chunk'].values)):
    print(zid)
    mol2_list.append(parse_chunk(chunk, zid))

mols = []
names = []
for mol2 in mol2_list:
    for _, mol, name in mol2.itertuples():
        mols.append(mol)
        names.append(name)

data_dict = {'mol': mols, 'mol_name': names}
mol_df_final = opd.ChemDataFrame(data=data_dict)

mol_df_final.to_mol2(osp.join(OUTDIR, 'experimental_ds.mol2'))