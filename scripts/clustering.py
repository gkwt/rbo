import sys, os

WORK_DIR = ".."
sys.path.append(WORK_DIR)

import pickle
import glob
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import norm
import scipy

import rdkit.Chem as Chem
from rbbo.utils import get_fingerprint

# plotting parameters
sns.set_theme(style="white", font_scale=1.5)
sns.set_palette("colorblind")
plt.rc("lines", linewidth=2.0, markersize=8)

datafile_name = 'ROGI'

def self_tanimoto_similarity(fingerprint_matrix: np.ndarray) -> np.ndarray:
    fp_matrix = fingerprint_matrix.astype(np.float64)
    intersections = fp_matrix @ fp_matrix.T
    popcounts = fp_matrix.sum(axis=1)
    unions = popcounts[:, np.newaxis] + popcounts[np.newaxis, :] - intersections

    with np.errstate(divide='ignore', invalid='ignore'):
        similarities = intersections / unions
    
    similarities[unions == 0] = 1.0
    
    return similarities[np.triu_indices(fp_matrix.shape[0], k=1)]


fnames = glob.glob(f'results/{datafile_name}/dat_files/*/results_*.pkl')
# fnames += glob.glob(f'../results/{datafile_name}/dat_files/*/results_ranking*.pkl')

n_start = 10
all_results = []
for f in tqdm(fnames, desc="Processing calibration files"):
    # info from filename
    parts = f.split('/')[-2].split('_')
    dataset_name = parts[0]
    target = parts[1]
    model = parts[-2]
    acq = parts[-1]
    res = pickle.load(open(f, 'rb'))
    loss_type = 'mse' if 'mse' in f else 'ranking'
    
    avg_internal_sim = {k: [] for k in ['chunk_sim', 'run']}
    for run in range(len(res)):
        df = res[run][0].iloc[n_start:]
    
        if model == 'gnn':
            smi_list = [g.smiles for g in df['feature']]
            fps = np.array([get_fingerprint(g.smiles) for g in df['feature']])
        else:
            fps = np.array(df['feature'].tolist())

        # Break fps into chunks of 5
        chunk_size = 10
        fps_chunks = [fps[i:i + chunk_size] for i in range(0, len(fps), chunk_size)]
        fps_chunks = [fps]
        
        # Calculate average pairwise distance for each chunk
        chunk_similarities = []
        for chunk in fps_chunks:
            # Calculate pairwise distances within chunk
            n_mols = len(chunk)
            pairwise_similarity = self_tanimoto_similarity(chunk)
            
            avg_sim = np.mean(pairwise_similarity)
            chunk_similarities.append(avg_sim)
        
        overall_avg_sim = np.mean(chunk_similarities)
    
        avg_internal_sim['chunk_sim'].append(overall_avg_sim)
        avg_internal_sim['run'].append(run)

    avg_internal_sim = pd.DataFrame(avg_internal_sim)
    avg_internal_sim['model'] = model
    avg_internal_sim['dataset_name'] = dataset_name
    avg_internal_sim['acq_func'] = acq
    avg_internal_sim['target'] = target
    avg_internal_sim['loss_type'] = loss_type

    all_results.append(avg_internal_sim)

all_results = pd.concat(all_results)

grouped_results = all_results.groupby(
    ['dataset_name', 'target', 'model', 'loss_type', 'acq_func']
).agg({'chunk_sim': ['mean', 'std']}).reset_index()     # this is done over the runs
grouped_results.columns = ['dataset_name', 'target', 'model', 'loss_type', 'acq_func', 'mean_chunk_sim', 'std_chunk_sim']


if dataset_name == 'zinc':
    grouped_results.to_csv(f'{dataset_name}_batch_internal_sim.csv', index=False)
else:
    grouped_results.to_csv(f'{target}_batch_internal_sim.csv', index=False)




    


