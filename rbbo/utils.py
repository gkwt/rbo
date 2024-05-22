import pandas as pd
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt
import torch.nn as nn

import numpy as np
import random
import torch

from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity, BulkTanimotoSimilarity
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

fpgen = AllChem.GetMorganGenerator(radius=3, includeChirality=True)



ORACLE_NAMES = ['QED', 'LogP', 'Celecoxib_Rediscovery', 'Aripiprazole_Similarity', 'Median_1', 
                'Osimertinib_MPO', 'Fexofenadine_MPO', 'Ranolazine_MPO', 'Perindopril_MPO', 'Amlodipine_MPO', 'Zaleplon_MPO',
                'Scaffold_Hop']

ORACLE_OBJ = {
    'QED': 'maximize',
    'LogP': 'minimize',
    'Celecoxib_Rediscovery': 'maximize', 
    'Aripiprazole_Similarity': 'maximize', 
    'Median_1': 'maximize', 
    'Osimertinib_MPO': 'maximize', 
    'Fexofenadine_MPO': 'maximize', 
    'Ranolazine_MPO': 'maximize', 
    'Perindopril_MPO': 'maximize', 
    'Amlodipine_MPO': 'maximize', 
    'Zaleplon_MPO': 'maximize',
    'Scaffold_Hop': 'maximize'
}

ORACLE_ROGI = {
    'Median_1': 0.019383141760157707, 
    'Aripiprazole_Similarity': 0.026202927416475286, 
    'LogP': 0.027440795306028776, 
    'Celecoxib_Rediscovery': 0.02827629458369757, 
    'Scaffold_Hop': 0.031596415488215734, 
    'Ranolazine_MPO': 0.040120983148615896, 
    'QED': 0.047706787639895054, 
    'Perindopril_MPO': 0.0654110006929221, 
    'Zaleplon_MPO': 0.06598271341542344, 
    'Amlodipine_MPO': 0.0722254608234878, 
    'Osimertinib_MPO': 0.07581081346260388, 
    'Fexofenadine_MPO': 0.07813300015970226
}


CHEMBL_KI = [
    'CHEMBL204_Ki', 'CHEMBL214_Ki', 'CHEMBL228_Ki', 'CHEMBL231_Ki', 
    'CHEMBL236_Ki', 'CHEMBL237_Ki', 'CHEMBL238_Ki', 'CHEMBL262_Ki',
    'CHEMBL264_Ki', 'CHEMBL287_Ki', 'CHEMBL1871_Ki', 'CHEMBL2034_Ki', 
    'CHEMBL2147_Ki', 'CHEMBL2835_Ki', 'CHEMBL2971_Ki', 'CHEMBL4005_Ki', 
    'CHEMBL4203_Ki', 'CHEMBL4792_Ki', 'CHEMBL1862_Ki',
    'CHEMBL219_Ki', 'CHEMBL233_Ki', 'CHEMBL234_Ki', 'CHEMBL244_Ki', 
]

CHEMBL_EC50 = [
    'CHEMBL218_EC50', 'CHEMBL235_EC50', 'CHEMBL237_EC50', 'CHEMBL239_EC50',
    'CHEMBL2047_EC50', 'CHEMBL3979_EC50', 'CHEMBL4616_EC50',
]

CHEMBL_ROGI = {
    'CHEMBL204_Ki': 0.005965208520326365, 
    'CHEMBL214_Ki': 0.02206337164214331, 
    'CHEMBL228_Ki': 0.030719315438331396, 
    'CHEMBL231_Ki': 0.06217269066499606, 
    'CHEMBL236_Ki': 0.04209749802611812, 
    'CHEMBL237_Ki': 0.02956707002080347, 
    'CHEMBL238_Ki': 0.03099899541800584, 
    'CHEMBL262_Ki': 0.026101448175572306, 
    'CHEMBL264_Ki': 0.015803424482213155, 
    'CHEMBL287_Ki': 0.046577439924283356, 
    'CHEMBL1871_Ki': 0.04004552059186896, 
    'CHEMBL2034_Ki': 0.02860126947829353, 
    'CHEMBL2147_Ki': 0.02171399941237334, 
    'CHEMBL2835_Ki': 0.037420742327794215, 
    'CHEMBL2971_Ki': 0.025451588992779134, 
    'CHEMBL4005_Ki': 0.036496755541569725, 
    'CHEMBL4203_Ki': 0.031133750514614894, 
    'CHEMBL4792_Ki': 0.08743653970206094, 
    'CHEMBL1862_Ki': 0.024711452879059848,
    'CHEMBL219_Ki': 0.012473008660025764,
    'CHEMBL233_Ki': 0.032246458399610045,
    'CHEMBL234_Ki': 0.02989212265679239,
    'CHEMBL244_Ki': 0.05996705242259187,
    'CHEMBL218_EC50': 0.035189428200530906, 
    'CHEMBL235_EC50': 0.04120365694078032, 
    'CHEMBL237_EC50': 0.03393567153559238, 
    'CHEMBL239_EC50': 0.05098602202792729, 
    'CHEMBL2047_EC50': 0.04493904890958536, 
    'CHEMBL3979_EC50': 0.033272785677484984, 
    'CHEMBL4616_EC50': 0.0363298992175774
}


def min_max_scale(x: pd.Series, min_val: float = None, max_val: float = None):
    if not min_val or not max_val:
        return minmax_scale(x)
    scaled_x = (x - min_val) / (max_val - min_val)
    return scaled_x

def remove_outliers(df: pd.DataFrame, goal: str = 'maximize', num_sigma: float = 3.0):
    mu = df['target'].mean()
    std = df['target'].std()

    if goal == 'maximize':
        df = df[df['target'] > mu - num_sigma*std]
    elif goal == 'minimize':
        df = df[df['target'] < mu + num_sigma*std]
    return df

def get_split_indices(num_runs, n_procs):
    indices = list(range(num_runs))
    chunk_size, remaining = divmod(num_runs, n_procs)
    split_inds = []
    for i in range(n_procs):
        start = i * chunk_size + min(i, remaining)
        end = (i + 1) * chunk_size + min(i + 1, remaining)
        split_inds.append(indices[start:end])
    return split_inds

def get_loss_function(loss_fn: str):
    if loss_fn == 'mse':
        return nn.MSELoss()
    elif loss_fn == 'ranking':
        return nn.MarginRankingLoss()#(margin=1.0)
    else:
        raise ValueError('Invalid loss_fn name.')


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def read_csv_float(filename):
    tmp = pd.read_csv(filename, nrows=5)
    float_cols = [c for c in tmp if tmp[c].dtype == "float64"]
    float32_cols = {c: np.float32 for c in float_cols}
    df = pd.read_csv(filename, engine='c', dtype=float32_cols)
    return df

def get_fingerprint(smi):
    mol = AllChem.MolFromSmiles(smi)
    fp = fpgen.GetFingerprint(mol)
    return fp

def average_dataset_diversity(smiles, num_comparisons=0):
    # select a few pairs from the dataset
    pairs = np.array(np.triu_indices(len(smiles), k=1, m=len(smiles))).transpose()
    if num_comparisons >= 0:
        if num_comparisons == 0:
            num_comparisons = len(smiles)
        pairs = pairs[np.random.choice(pairs.shape[0], num_comparisons, replace=False), :]

    tot = 0
    for p in pairs:
        fp1 = get_fingerprint(smiles[p[0]])
        fp2 = get_fingerprint(smiles[p[1]])
        tot += TanimotoSimilarity(fp1, fp2)
    return tot / pairs


def read_hparam_files(fname: str):
    hparams = {}
    with open(fname, 'r') as f:
        for l in f:
            if ':' in l:
                l = l.strip().split(':')
                hparams[l[0]] = l[1]
    return hparams
