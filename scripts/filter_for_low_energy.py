import pandas as pd
import numpy as np
import oddt.pandas as opd
import sys

def low_energy_filter(mol_file, out_file):
    mol2_df = opd.read_mol2(mol_file, skip_bad_mols=True)
    mol2_df['best_energy'] = mol2_df.groupby(['mol_name'])['Total Energy'].transform(min)
    mol2_df = mol2_df[mol2_df['Total Energy'] == mol2_df['best_energy']]
    mol2_df.drop(columns='best_energy')
    mol2_df.to_mol2(out_file)


if __name__ == "__main__":
    mol_file = sys.argv[1]
    out_file = sys.argv[2]
    low_energy_filter(mol_file, out_file)