import os
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch import optim
from torch_geometric.loader import DataLoader as pygdl
from torch.utils.data import DataLoader as dl

import gpytorch

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

from rbbo.data import DataframeDataset, PairwiseRankingDataframeDataset
from rbbo.models import MLP, BNN, GP, GNN
from rbbo.early_stop import EarlyStopping
from rbbo import utils


class BayesOptCampaign:
    """
    Args:
        dataset (pd.DataFrame)
        goal (str): optimization goal, 'maximize' or 'minimize'
        model
        acq_func_type (str): acquisition function type
        num_acq_samples (int): number of samples drawn in each round of acquisition function optimization
        batch_size (int): number of recommendations provided by the acquisition function at each
            iteration
        budget (int): maximum tolerated objective function measurements for a single
            optimization campaign
        init_design_strategy (str): strategy used to select the initial design points
        num_init_design (int): number of initial design points to propose
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        goal: str,
        loss_type: str,
        acq_func: Callable,
        model_type: str = 'mlp',
        num_of_epochs: int = 100,
        num_total: int = 10,
        batch_size: int = 1,
        budget: int = 100,
        init_design_strategy: str = "random",
        num_init_design: int = 20,
        work_dir: str = ".",
        verbose: bool = True,
        num_acq_samples: int = -1,
        use_gpu: bool = True,
        scale: bool = False,
        test_ratio: float = 0.15,
        learning_rate: float = 0.005, 
        *args,
        **kwargs,
    ):
        self.dataset = dataset
        self.goal = goal
        self.model_type = model_type
        self.acq_func = acq_func
        self.batch_size = batch_size
        self.budget = budget
        self.init_design_strategy = init_design_strategy
        self.work_dir = work_dir
        self.num_init_design = num_init_design
        self.loss_type = loss_type
        self.loss_func = utils.get_loss_function(loss_type)
        self.num_of_epochs = num_of_epochs
        self.num_total = num_total
        self.verbose = verbose
        self.num_acq_samples = num_acq_samples
        self.scale = scale
        self.test_ratio = test_ratio
        self.lr = learning_rate

        if use_gpu:
            if torch.cuda.is_available():
                print('GPU found, and is used.')
                self.use_gpu = True
                self.device = torch.device('cuda')
            else:
                print('No GPU found, default to CPU.')
                self.use_gpu = False
                self.device = torch.device('cpu')
        else:
            self.use_gpu = False
            self.device = torch.device('cpu')

        # decide if we are using PyG or just vector features
        self.use_graphs = self.model_type == 'gnn'

        # create a split if required
        # the split is always the same random_state (ensuring same test set across runs)
        if test_ratio > 0:
            self.eval_mode = True
            self.entire_df, self.held_df = train_test_split(self.dataset, test_size=self.test_ratio, random_state=42)
        else:
            self.eval_mode = False
            self.entire_df = self.dataset

        self._validate_budget()

        # create working dir, and write the hypermarameters
        os.makedirs(self.work_dir, exist_ok=True)
        with open(self.work_dir + '/hparams.txt', 'w') as f: 
            for k,v in self.__dict__.items():
                f.write(f'{k}: {v}\n')

    def _validate_budget(self):
        """validate that the budget does not exceed the total number of
        options in the dataset
        """
        if self.budget + self.num_init_design > len(self.entire_df):
            raise ValueError(f'There are only {len(self.entire_df)} in the dataset. \
            Requested budget and initial design is {self.budget + self.num_init_design }. Exiting...')

    def _reinitialize_model(self, meas_df = None):
        """Reinitialize the model from scratch (re-train)"""
        if self.model_type == 'mlp':
            reinit_model = MLP()
        elif self.model_type == 'bnn':
            reinit_model = BNN()
        elif self.model_type == 'gnn':
            gr = meas_df['feature'].tolist()[0]
            reinit_model = GNN(
                num_node_features = gr.x.shape[-1],
                num_edge_features = gr.edge_attr.shape[-1],
            )
        elif self.model_type == 'gp':
            assert meas_df is not None, 'GPs require the training data for initialization'
            x = torch.tensor(np.array(meas_df['feature'].tolist(), dtype=np.float32))
            y = torch.tensor(np.array(meas_df['target'].tolist(), dtype=np.float32))
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            reinit_model = GP(x, y, likelihood)
            self.loss_func = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, reinit_model)     # use mll for GPs
        else:
            raise ValueError('No such model specified.')
        return reinit_model

    def run(self, seed: int = 42):
        """Run the sequential learning experiments for independently seeded executions.
        seed: int           specifies seed of all random selection algorithms
        eval: bool          evaluating the surrogate on a test split
        """

        # set seed for reproducibility
        utils.set_seed(seed)

        observations = []
        predictions = {key: [] for key in ['iteration', 'y_true', 'y_pred', 'y_std']}
        iter_num = 0
        eval_num = 0 

        while (len(observations) - self.num_init_design) < self.budget:
            # complete all init design first
            if iter_num == 0:
                avail_df = self.entire_df
                for _ in range(self.num_init_design):
                    df_samp = self.sample_meas_randomly(avail_df)
                    sample, measurement, feature = df_samp['smiles'], df_samp['target'], df_samp['feature']
                    observations.append(
                        {
                            "evaluation": eval_num,
                            "smiles": sample,
                            "target": measurement,
                            "feature": feature,
                        }
                    )

                    # update avail_df that have been sampled already
                    _, avail_df = self.split_avail(avail_df, observations)
                    eval_num += 1

            # split dataset into measured and available candidates
            meas_df, avail_df = self.split_avail(self.entire_df, observations)

            # perform a 90/10 split for early stopping
            meas_df_train, meas_df_val = train_test_split(meas_df, test_size=0.1, random_state=seed)

            if self.scale:
                scaler = RobustScaler()     # this works best with both Gaussian distr targets, and targets with outliers
                meas_df_train['target'] = scaler.fit_transform(meas_df_train[['target']])
                meas_df_val['target'] = scaler.transform(meas_df_val[['target']])
                meas_df['target'] = scaler.transform(meas_df[['target']])
            # shuffle the available candidates (acq func sampling)
            if self.num_acq_samples > 0 and len(avail_df) > self.num_acq_samples:
                avail_df = avail_df.sample(n=self.num_acq_samples).reset_index(drop=True)
            if self.verbose:
                print(
                    f"NUM_ITER : {iter_num}\tNUM OBS : {len(observations)}"
                )

            # re-initialize the surrogate model
            model = self._reinitialize_model(meas_df_train)
            model = model.to(self.device)
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=self.lr,
            )
            es = EarlyStopping(model, mode='minimize', patience=25)

            # convert X_meas and y_meas to torch Dataset
            if self.loss_type == 'mse':
                meas_train = DataframeDataset(meas_df_train)
                meas_val = DataframeDataset(meas_df_val)
            elif self.loss_type == 'ranking':
                meas_ds = PairwiseRankingDataframeDataset(meas_df)
                meas_train, meas_val = torch.utils.data.random_split(meas_ds, [0.9, 0.1], torch.Generator().manual_seed(seed))
            avail_set = DataframeDataset(avail_df)

            # load data use DataLoader
            DataLoader = pygdl if self.use_graphs else dl
            meas_train = DataLoader(meas_train, batch_size=128, shuffle=True)
            meas_val = DataLoader(meas_val, batch_size=64, shuffle=False)
            avail_loader = DataLoader(avail_set, batch_size=512, shuffle=False)

            # train the model on observations
            # start with fresh model every time
            # specify the LOSS function -> (ranking/mse)
            for epoch in range(self.num_of_epochs):
                model.train()
                for data in meas_train:
                    self.optimizer.zero_grad()
                    loss = model.step(
                        data, 
                        loss_type=self.loss_type, 
                        loss_func=self.loss_func,
                        device=self.device
                    )
                    loss.backward()
                    self.optimizer.step()

                # evaluate on validation set for early stopping
                model.eval()
                running_loss = 0
                with torch.no_grad():
                    for data in meas_val:
                        loss = model.step(
                            data, 
                            loss_type=self.loss_type, 
                            loss_func=self.loss_func,
                            device=self.device
                        )
                        running_loss += loss.detach().cpu().numpy()
                    running_loss /= len(meas_val)

                stop = es.check_criteria(running_loss, model)
                if stop:
                    # end training
                    break

            # restore the best model
            model.load_state_dict(es.restore_best())


            # make inference
            mu_avail, std_avail = [], []
            with torch.no_grad():
                model.eval()
                for data in avail_loader:
                    X_avail, _ = data
                    X_avail = X_avail.to(self.device)
                    y_avail, y_var = model.predict(X_avail)
                    mu_avail.append(y_avail.detach().cpu().numpy())
                    if y_var is not None:
                        std_avail.append(torch.sqrt(y_var).detach().cpu().numpy())

            # gather results for acqusition function evaluation
            mu_avail = np.concatenate(mu_avail).flatten()
            if not std_avail:
                std_avail = np.array([None]*len(mu_avail))
            else:
                std_avail = np.concatenate(std_avail).flatten()

            # calculate acq function
            # negate the results of prediction if minimizing
            if self.goal == "minimize":
                mu_avail *= -1.0
                y_best = -meas_df['target'].min()    # calculate the min and negate
            elif self.goal == "maximize":
                y_best = meas_df['target'].max()
            else:
                raise ValueError('Goal must be minimize or maximize.')
            acq_vals = self.acq_func(mu_avail, std_avail, best_val=y_best)   

            # higher acq_vals the better
            sort_idxs = np.argsort(acq_vals)[::-1]  # descending order
            sample_idxs = sort_idxs[: self.batch_size]

            # gather predictions for later analysis
            # if test_ratio is specified
            if self.eval_mode:
                held_dl = DataLoader(
                    DataframeDataset(self.held_df), batch_size=128, shuffle=False
                )
                mu, std, y_true = [], [], []
                with torch.no_grad():
                    model.eval()
                    for data in held_dl:
                        X_avail, y = data
                        X_avail = X_avail.to(self.device)
                        y_avail, y_var = model.predict(X_avail)
                        mu.append(y_avail.detach().cpu().numpy())
                        y_true.append(y.numpy())
                        if y_var is not None:
                            std.append(torch.sqrt(y_var).detach().cpu().numpy())
                        else:
                            std.append(np.array([np.nan]*len(y_avail)))

                mu = np.concatenate(mu).flatten()
                std = np.concatenate(std).flatten()
                y_true = np.concatenate(y_true).flatten()
                if self.scale:
                    y_pred = scaler.inverse_transform(mu.reshape(-1,1)).flatten()
                else:
                    y_pred = mu
                
                predictions['iteration'].extend([iter_num]*len(y_true))
                predictions['y_true'].extend(y_true.tolist())
                predictions['y_pred'].extend(y_pred.tolist())
                predictions['y_std'].extend(std.tolist())

            # perform measurements
            for sample_idx in sample_idxs:
                df_samp = self.sample_meas_acq(
                    avail_df, sample_idx
                )
                sample, measurement, feature = df_samp['smiles'], df_samp['target'], df_samp['feature']

                observations.append(
                    {"evaluation": eval_num, "smiles": sample, "target": measurement, "feature": feature}
                )
                eval_num += 1

            iter_num += 1

        if self.eval_mode:
            return pd.DataFrame(observations), pd.DataFrame(predictions)
        else:
            return pd.DataFrame(observations)
        

    @staticmethod
    def sample_meas_acq(avail_df, idx):
        """obtain the molecules suggested by the acquisition function"""
        return avail_df.iloc[idx]

    @staticmethod
    def sample_meas_randomly(avail_df):
        """take a single random sample from the available candiates"""
        idx = np.random.randint(avail_df.shape[0])
        return avail_df.iloc[idx]

    @staticmethod
    def split_avail(data, observations):
        """return available and measured datasets"""
        obs_smi = [o["smiles"] for o in observations]

        # avail_df is the set of molecules that have not been measured
        # create a function that checks if the smiles is in
        avail_df = data[~(data["smiles"].isin(obs_smi))]
        meas_df = data[data["smiles"].isin(obs_smi)]
        return meas_df, avail_df

