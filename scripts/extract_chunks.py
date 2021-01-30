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
for zid, chunk in tqdm(zip(exp_df['ZINC ID'], exp_df['chunk'])):
    print(zid)
    mol2_list.append(parse_chunk(chunk, zid))



#full_mol2 = mol2_list[0]
full_mol2 = pd.concat(mol2_list)
print(full_mol2)

full_mol2.to_mol2(osp.join(OUTDIR, 'experimental_ds.mol2'))
