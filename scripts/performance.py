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

from scipy.stats import kendalltau, spearmanr, pearsonr
from sklearn.metrics import r2_score
from sklearn.preprocessing import RobustScaler

# plotting parameters
sns.set_theme(style="white", font_scale=1.5)
sns.set_palette("colorblind")
plt.rc("lines", linewidth=2.0, markersize=8)


def get_statistics(series):
    mean = series.mean()
    sem = series.sem()
    std = series.std()
    ci95_lower = series.quantile(0.025)
    ci95_upper = series.quantile(0.975)
    return mean, std, sem, ci95_lower, ci95_upper


for datafile_name in ['ROGI', 'EC50', 'KI']:
    for check_iter in [0, 49, 99]:

        # ---------- PAIRWISE RANKING ----------
        fnames = glob.glob(f'results/{datafile_name}/dat_files/*_gnn_*/results_ranking*.pkl')
        fnames += glob.glob(f'results/{datafile_name}/dat_files/*_bnn_*/results_ranking*.pkl')
        fnames += glob.glob(f'results/{datafile_name}/dat_files/*_mlp_*/results_ranking*.pkl')

        all_results = []
        for f in tqdm(fnames, desc=f"Processing ranking calibration files for {datafile_name} at iter {check_iter}"):
            res = pickle.load(open(f, 'rb'))

            save_dir = f.split('/dat_files')[0] + '/calibration_ranking'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            
            plot_df = []
            corr_stats = []
            for i in range(len(res)):
                df = res[i][1]
                tmp_df = df[df['iteration'] == check_iter].copy()
                if tmp_df.empty:
                    continue

                # ranking models do not need to be unscaled, the scores are as they are
                if datafile_name in ['KI', 'EC50']:
                    meas_df = res[i][0]
                    scaler = RobustScaler()
                    scaler.fit(meas_df[["target"]].to_numpy())
                    y_pred_scaled = scaler.transform(tmp_df[["y_pred"]].to_numpy()).squeeze().tolist()
                    tmp_df["y_pred"] = y_pred_scaled

                # Calculate Kendall tau on the ground truth for this run
                # Use y_pred and y_true from the selected iteration
                y_true = tmp_df['y_true'].to_numpy()
                y_pred = tmp_df['y_pred'].to_numpy()

                kendall, kendall_pval = kendalltau(y_pred, y_true)
                spearman, spearman_pval = spearmanr(y_pred, y_true)
                corr_stats.append({
                    "run": i,
                    "kendall": kendall,
                    "kendall_pval": kendall_pval,
                    "spearman": spearman,
                    "spearman_pval": spearman_pval
                })

            if not corr_stats:
                continue

            corr_stats = pd.DataFrame(corr_stats)

            # info from filename
            parts = f.split('/')[-2].split('_')
            dataset_name = parts[0]
            target = parts[1]
            model = parts[-2]
            acq = parts[-1]

            kendall, kendall_std, kendall_sem, kendall_95ci_lower, kendall_95ci_upper = get_statistics(corr_stats["kendall"])
            spearman, spearman_std, spearman_sem, spearman_95ci_lower, spearman_95ci_upper = get_statistics(corr_stats["spearman"])
            
            # Create dataframe with results, including 95% confidence intervals
            results_df = pd.DataFrame({
                'dataset_name': [dataset_name],
                'target': [target],
                'model': [model],
                'acq': [acq],
                'kendall': [kendall],
                'kendall_std': [kendall_std],
                'kendall_sem': [kendall_sem],
                'kendall_95ci_lower': [kendall_95ci_lower],
                'kendall_95ci_upper': [kendall_95ci_upper],
                'spearman': [spearman],
                'spearman_std': [spearman_std],
                'spearman_sem': [spearman_sem],
                'spearman_95ci_lower': [spearman_95ci_lower],
                'spearman_95ci_upper': [spearman_95ci_upper],
            })
            all_results.append(results_df)

        if all_results:
            pd.concat(all_results, ignore_index=True).sort_values(['dataset_name', 'model', 'acq']).to_csv(
                f'{save_dir}/performance_results_iter{check_iter}.csv', index=False
            )
        


        # ---------- REGRESSION ----------

        # for regression and GP
        fnames = glob.glob(f'results/{datafile_name}/dat_files/*_gnn_*/results_mse*.pkl')
        fnames += glob.glob(f'results/{datafile_name}/dat_files/*_bnn_*/results_mse*.pkl')
        fnames += glob.glob(f'results/{datafile_name}/dat_files/*_mlp_*/results_mse*.pkl')
        fnames += glob.glob(f'results/{datafile_name}/dat_files/*_gp_*/results_mse*.pkl')

        regression_results = []

        for f in tqdm(fnames, desc=f"Processing regression calibration files for {datafile_name} at iter {check_iter}"):
            res = pickle.load(open(f, "rb"))

            save_dir = f.split('/dat_files')[0] + '/calibration_regression'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)

            corr_stats = []
            for i in range(len(res)):
                df = res[i][1]
                tmp_df = df[df['iteration'] == check_iter].copy()
                if tmp_df.empty:
                    continue

                # if scaling was used, transform the uncertainty
                if datafile_name in ['KI', 'EC50']:
                    meas_df = res[i][0]
                    scale = RobustScaler().fit(meas_df[["target"]]).scale_[0]
                    tmp_df['y_std'] = tmp_df["y_std"] * scale

                y_true = tmp_df['y_true'].to_numpy()
                y_pred = tmp_df['y_pred'].to_numpy()

                kendall, kendall_pval = kendalltau(y_true, y_pred)
                spearman, spearman_pval = spearmanr(y_true, y_pred)
                pearson, pearson_pval = pearsonr(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)
                corr_stats.append({
                    "run": i,
                    "kendall": kendall,
                    "kendall_p": kendall_pval,
                    "spearman": spearman,
                    "spearman_p": spearman_pval,
                    "pearson": pearson,
                    "pearson_p": pearson_pval,
                    "r2": r2,
                })
            
            if not corr_stats:
                continue

            corr_stats = pd.DataFrame(corr_stats)
            
            # info from filename
            parts = f.split('/')[-2].split('_')
            dataset_name = parts[0]
            target = parts[1]
            model = parts[-2]
            acq = parts[-1]
            
            # Calculate statistics for correlation metrics across runs (mean and 95% CI for kendall_tau, spearman, pearson, r2)
            kendall_mean, kendall_std, kendall_sem, kendall_95ci_lower, kendall_95ci_upper = get_statistics(corr_stats['kendall'])
            spearman_mean, spearman_std, spearman_sem, spearman_95ci_lower, spearman_95ci_upper = get_statistics(corr_stats['spearman'])
            pearson_mean, pearson_std, pearson_sem, pearson_95ci_lower, pearson_95ci_upper = get_statistics(corr_stats['pearson'])
            r2_mean, r2_std, r2_sem, r2_95ci_lower, r2_95ci_upper = get_statistics(corr_stats['r2'])
            
            # Create dataframe with results
            results_df = pd.DataFrame({
                'dataset_name': [dataset_name],
                'target': [target],
                'model': [model],
                'acq': [acq],
                'kendall': [kendall_mean],
                'kendall_std': [kendall_std],
                'kendall_sem': [kendall_sem],
                'kendall_95ci_lower': [kendall_95ci_lower],
                'kendall_95ci_upper': [kendall_95ci_upper],
                'spearman': [spearman_mean],
                'spearman_std': [spearman_std],
                'spearman_sem': [spearman_sem],
                'spearman_95ci_lower': [spearman_95ci_lower],
                'spearman_95ci_upper': [spearman_95ci_upper],
                'pearson': [pearson_mean],
                'pearson_std': [pearson_std],
                'pearson_sem': [pearson_sem],
                'pearson_95ci_lower': [pearson_95ci_lower],
                'pearson_95ci_upper': [pearson_95ci_upper],
                'r2': [r2_mean],
                'r2_std': [r2_std],
                'r2_sem': [r2_sem],
                'r2_95ci_lower': [r2_95ci_lower],
                'r2_95ci_upper': [r2_95ci_upper],
            })
            regression_results.append(results_df)

        if regression_results:
            pd.concat(regression_results, ignore_index=True).sort_values(['dataset_name', 'model', 'acq']).to_csv(
                f'{save_dir}/performance_results_iter{check_iter}.csv', index=False
            )
