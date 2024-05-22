import torch
from torch.utils.data import Dataset

import multiprocessing
from functools import partial

import numpy as np
import pandas as pd
import rdkit.Chem.AllChem as Chem

import torch_geometric as pyg
from torch_geometric.utils import from_smiles

from .mpnn import RevIndexedData
from .utils import read_csv_float

# import mordred
# import mordred.descriptors
# calc = mordred.Calculator(mordred.descriptors, ignore_3D=True)

# def get_mordred_features(smiles: str):
#     m = Chem.MolFromSmiles(smiles)
#     if m is None:
#         return None
#     return calc(m)._values


def get_fingerprint(smiles: str):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    return np.array(Chem.GetMorganFingerprintAsBitVect(m, 3), dtype=np.float32)


def process_dataset(smiles_data: str, feature_type: str, num_workers: int = 1):
    data = pd.read_csv(smiles_data)
    data = data.dropna()

    if feature_type == "mordred":
        raise ValueError('Mordred temporarily not available.')
        pass
        # with multiprocessing.Pool(num_workers) as pool:
        #     desc = pool.map(get_mordred_features, self.smiles.tolist())
        # desc = np.array(desc, dtype=float)
        # desc = desc[:, ~np.isnan(desc).any(axis=0)]

        # self.data["feature"] = list(desc)
        # self.data = self.data.dropna(axis=1)
        # self.feature = self.data["feature"]

    elif feature_type == "fp":
        with multiprocessing.Pool(num_workers) as pool:
            fps = pool.map(get_fingerprint, data["smiles"].tolist())
        data["feature"] = fps
    
    elif feature_type == "graph":
        # multiprocessing has issues with pickling results
        # do serially for now
        gr = []
        for smi in data["smiles"]:
            gr.append(RevIndexedData(from_smiles(smi)))
        data['feature'] = gr

    else:
        # if features is not defined, just use the smiles
        data['feature'] = data["smiles"]

    data["target"] = data["target"].astype('float32')

    return data


class DataframeDataset(Dataset):
    """
    Quick wrapper to create a dataset for pytorch training
    directly from dataframe.
    Requires a column named "feature" and one named "target"
    """

    def __init__(self, df: pd.DataFrame):
        self.data = df
        assert "feature" in df.columns, 'Column for "feature" not found.'
        assert "target" in df.columns, 'Column for "target" not found.'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data["feature"].iloc[idx], torch.tensor(self.data["target"].iloc[idx], dtype=torch.float32)

class PairwiseRankingDataframeDataset(Dataset):
    """
    This dataset will load the data for pairwise ranking loss.
    """
    def __init__(self, df: pd.DataFrame, max_num_pairs: int = 0):
        self.data = df
        assert "feature" in df.columns, 'Column for "feature" not found.'
        assert "target" in df.columns, 'Column for "target" not found.'

        self.max_num_pairs = max_num_pairs
        n = len(df)

        # default to length of dataframe
        if self.max_num_pairs == 0:
            self.max_num_pairs = 2*n

        if self.max_num_pairs > (n**2 - n)/2:
            self.max_num_pairs = int((n**2 - n)/2)
         
        # the ranking based on the target value
        self.compare_fn = np.greater

        # get indices for pairs
        # will produce (n^2-n)/2 data pairs, which can be truncated
        pairs = np.array(np.triu_indices(len(df), k=1, m=len(df))).transpose()
        
        if max_num_pairs >= 0:
            self.pairs = pairs[np.random.choice(pairs.shape[0], self.max_num_pairs, replace=False), :]
        else:
            self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        # m(x1) > m(x2) is y = +1
        # m(x1) < m(x2) is y = -1
        idx = self.pairs[idx]
        target = (self.data.iloc[idx[0]].target > self.data.iloc[idx[1]].target).astype(np.float32)
        target = target * 2.0 - 1.0

        return self.data.iloc[idx[0]]['feature'], self.data.iloc[idx[1]]['feature'], torch.tensor(target, dtype=torch.float32)



    



