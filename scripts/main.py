import sys
sys.path.append("..")

import multiprocessing
import torch
from functools import partial
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm, kendalltau

from sklearn.metrics import r2_score

from tqdm import tqdm

from rbbo.data import process_dataset
from rbbo.bo import BayesOptCampaign

from argparse import ArgumentParser

def ucb(mu: np.array, sigma: np.array, beta: float = 0.3, **kwargs):
    if sigma is None:
        return mu
    return mu + beta*sigma

def greedy(mu: np.array, sigma: np.array, **kwargs):
    return mu

def ei(mu: np.array, sigma: np.array, best_val: float, **kwargs):
    z = (mu - best_val) / sigma
    ei = (mu - best_val) * norm.cdf(z) + sigma * norm.pdf(z)
    return ei

def get_acq_function(key: str):
    if key == 'ucb':
        return ucb
    elif key == 'greedy':
        return greedy
    elif key == 'ei':
        return ei
    else:
        raise ValueError('No such acq function implemented.')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", action="store", type=str, help="Dataset to study.")
    parser.add_argument("--num_workers", action="store", type=int, default=1, help="Number of workers, defaults 1.")
    parser.add_argument("--num_runs", action="store", type=int, dest="num_runs", help="Number of runs in BO. Defaults 20.", default=20)
    parser.add_argument("--num_init", action="store", type=int, dest="num_init", help="Number of initial samples defaults 20.", default=20)
    parser.add_argument("--budget", action="store", type=int, dest="budget", help="Budget of surrogate supported iterations. Defaults 100.", default=100)
    parser.add_argument("--batch_size", action="store", type=int, dest="batch_size", help="Batch per iteration. Defaults 1.", default=1)
    parser.add_argument("--num_epochs", action="store", type=int, dest="num_epochs", help="Number of epochs per iteration. Defaults 5.", default=5)
    parser.add_argument("--maximize", action="store_true", dest="goal", help="Set goal to maximize. Otherwise will minimize.", default=False)
    parser.add_argument("--rank", action="store_true", help="Toggle use of ranking loss. Otherwise, MSE.", default=False)
    parser.add_argument("--acq_func", action="store", type=str, dest="acq_func", help="Acqusition function used. Defaults to greedy.", default='greedy')
    parser.add_argument("--model_type", action="store", type=str, dest="model_type", help="Change deep model used. Defaults to mlp.", default='mlp')
    parser.add_argument("--use_gpu", action="store_true", help="Toggle use of gpu. Most useful for BNN. Defaults false.", default=False)
    parser.add_argument("--scale", action="store_true", help="Toggle scaling of targets. Defaults false.", default=False)
    parser.add_argument("--test_ratio", action="store", type=float, dest="test_ratio", help="Size of testing set, set to 0 for no evaluation. \
                        Defaults 0.15.", default=0.15)
    parser.add_argument("--learning_rate", action="store", type=float, dest="learning_rate", help="Setting learning rate, defaults to 0.005.", default=0.005)
    FLAGS = parser.parse_args()

    # input parameters
    dataset_name = FLAGS.dataset_name
    goal = 'maximize' if FLAGS.goal else 'minimize'
    loss_type = 'ranking' if FLAGS.rank else 'mse' 
    model_type = FLAGS.model_type
    num_workers = FLAGS.num_workers
    num_runs = FLAGS.num_runs
    num_init = FLAGS.num_init
    batch_size = FLAGS.batch_size
    budget = FLAGS.budget
    num_epochs = FLAGS.num_epochs
    acq_func = get_acq_function(FLAGS.acq_func)
    use_gpu = FLAGS.use_gpu
    scale = FLAGS.scale
    test_ratio = FLAGS.test_ratio
    learning_rate = FLAGS.learning_rate

    # other variables dependent on inputs
    work_dir = f'{dataset_name}_{goal}_{model_type}_{FLAGS.acq_func}'
    dataset_path = f'../data/{dataset_name}.csv'
    if model_type == 'gnn':
        dataset = process_dataset(dataset_path, "graph", num_workers=num_workers)
        print("GNN uses graph features.")
    else:
        dataset = process_dataset(dataset_path, "fp", num_workers=num_workers)
        print("Using ECFP fingerprints.")

    bo = BayesOptCampaign(
        dataset, 
        goal, 
        model_type = model_type,
        loss_type = loss_type, 
        acq_func = acq_func,
        num_of_epochs = num_epochs,
        budget = budget, 
        batch_size = batch_size,
        num_init_design = num_init, 
        verbose = False, 
        work_dir = work_dir,
        num_acq_samples = -1,      # consider changing this to a finite positive number (ie. 128) to speed up inference
        use_gpu = use_gpu,
        scale = scale,
        test_ratio = test_ratio,
        learning_rate = learning_rate,
    )

    # perform the run
    if num_runs > 1 and not use_gpu:       
        with multiprocessing.Pool(num_workers) as pool:
            bo_results = pool.map(bo.run, range(num_runs))
    else:
        print('Running sequentially because only singular run, or use of GPU.')
        bo_results = []
        for i in tqdm(range(num_runs)):
            bo_results.append(bo.run(i))

    pickle.dump(bo_results, open(f'{work_dir}/results_{loss_type}_{dataset_name}_{goal}.pkl', 'wb'))

