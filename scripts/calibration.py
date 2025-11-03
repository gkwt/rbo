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


def pairwise_calibration_curve(y_pred, y_std, y_true, n_bins=10):
    """
    Creates a pairwise calibration plot (reliability diagram) for ranking uncertainties.
    """
    y_pred, y_std, y_true = np.array(y_pred), np.array(y_std), np.array(y_true)
    
    predictions = []
    outcomes = []
    
    # Look at probability of P(i>j) triu, P(i<j) tril
    s_diff = y_pred.reshape(-1,1) - y_pred.reshape(-1,1).transpose()
    u_diff = np.sqrt(y_std.reshape(-1,1)**2 + y_std.reshape(-1,1).transpose()**2)  # error propagation
    u_diff += 1e-8      # to handle division by zeros
    z_score = s_diff / u_diff       # mean = 0, so i=j is 50% correct ranking
    prob = norm.cdf(z_score)        # if i > j, then z_score > 0
                                    # cdf gives the probability of P(i > j)
                                    # same for lower triangle and P(i < j)

    # binary outcome of what the true relvance is
    outcome = (y_true.reshape(-1,1) > y_true.reshape(-1,1).transpose()).astype(int)

    # Skip where true label is the same and or same i and j (since i != j)
    ignore_indices = y_true.reshape(-1,1) == y_true.reshape(-1,1).transpose()
    prediction = prob[~ignore_indices]
    outcome = outcome[~ignore_indices]

    outcomes.extend(outcome.tolist())
    predictions.extend(prediction.tolist())
            
    if not predictions:
        print("No valid pairs found to create a calibration plot.")
        return

    # Create reliability diagram histogram bins
    df = pd.DataFrame({'predicted_prob': predictions, 'outcome': outcomes})
    df['bin'] = pd.cut(df['predicted_prob'], bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)

    return df


def regression_calibration_curve(y_pred, y_std, y_true, n_bins=10):
    """
    Compute regression calibration curve using C(q) score.
    C(q) = fraction of true values within the predicted q-confidence interval.
    Returns a DataFrame with columns: 'quantile', 'cq', 'bin_lower', 'bin_upper'
    """
    y_pred, y_std, y_true = np.array(y_pred), np.array(y_std), np.array(y_true)
    quantiles = np.linspace(0, 1, n_bins+1)

    cqs = []
    counts = []
    for i, q in enumerate(quantiles):
        # For a normal distribution, the interval [-z, z] covers q fraction of the probability mass
        z = scipy.stats.norm.ppf(0.5 + q/2., loc=0., scale=1.)
        lower = y_pred - z * y_std
        upper = y_pred + z * y_std
        in_interval = (y_true > lower) & (y_true < upper)
        cqs.append(np.mean(in_interval))
        counts.append(np.sum(in_interval))

    df = pd.DataFrame({
        'quantile': quantiles,
        'cq': cqs,
        'counts': counts
    })
    df['bin'] = pd.cut(df['quantile'], bins=np.linspace(0, 1, n_bins + 2), include_lowest=True)
    return df


def get_statistics(series):
    mean = series.mean()
    sem = series.sem()
    std = series.std()
    ci95_lower = series.quantile(0.025)
    ci95_upper = series.quantile(0.975)
    return mean, std, sem, ci95_lower, ci95_upper


for datafile_name in ['KI', 'ROGI', 'EC50']:

    for check_iter in [0, 49, 99]:

        # ---------- PAIRWISE RANKING ----------
        fnames = glob.glob(f'results/{datafile_name}/dat_files/*_gnn_*/results_ranking*.pkl')
        fnames += glob.glob(f'results/{datafile_name}/dat_files/*_bnn_*/results_ranking*.pkl')

        all_results = []
        for f in tqdm(fnames, desc=f"Processing ranking calibration files for {datafile_name}, iter {check_iter}"):
            res = pickle.load(open(f, 'rb'))

            save_dir = f.split('/dat_files')[0] + '/calibration_ranking'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            
            plot_df = []
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

                calib_df = pairwise_calibration_curve(tmp_df['y_pred'].to_numpy(), tmp_df['y_std'].to_numpy(), tmp_df['y_true'].to_numpy(), n_bins=10)
                calib_df["run"] = i
                plot_df.append(calib_df)

            if not plot_df:
                continue

            plot_df = pd.concat(plot_df)

            # gather by bins
            calibration_data = plot_df.groupby(['bin']).agg(
                mean_predicted_prob=('predicted_prob', 'mean'),
                fraction_of_positives=('outcome', 'mean'),
                sem_fraction_of_positives=('outcome', 'sem'),
                count=('outcome', 'size')
            ).dropna()

            # info from filename
            parts = f.split('/')[-2].split('_')
            dataset_name = parts[0]
            target = parts[1]
            model = parts[-2]
            acq = parts[-1]

            # Plotting
            fig = plt.figure(figsize=(8, 8))
            plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
            
            plt.plot(calibration_data['mean_predicted_prob'], calibration_data['fraction_of_positives'], 's-', label='Model calibration')
            plt.fill_between(
                calibration_data['mean_predicted_prob'],
                calibration_data['fraction_of_positives'] - 1.96*calibration_data['sem_fraction_of_positives'],
                calibration_data['fraction_of_positives'] + 1.96*calibration_data['sem_fraction_of_positives'],
                alpha=0.2,
            )
            
            plt.title('Pairwise Ranking Calibration Plot')
            plt.xlabel('Average predicted probability')
            plt.ylabel('Fraction of correctly ordered pairs')
            plt.xlim([0,1])
            plt.ylim([0,1])
            plt.legend()
            plt.grid(True, alpha=0.5)
            plt.axis('square')
            fig.savefig(f'{save_dir}/calibration-curve_{dataset_name}_{target}_{model}_{acq}_iter{check_iter}.png')
            plt.close()


            # Calculate ECE (Expected Calibration Error) weighted by bin counts
            # gather by bins and runs
            # Make 'run' a column instead of an index after groupby
            calibration_data_run = (
                plot_df.groupby(['run', 'bin']).agg(
                    mean_predicted_prob=('predicted_prob', 'mean'),
                    fraction_of_positives=('outcome', 'mean'),
                    count=('outcome', 'size')
                ).dropna().reset_index()  # This makes 'run' and 'bin' columns instead of indices
            )
        
            # Calculate statistics for each run in calibration_data_run, then aggregate
            run_stats = []
            for run, run_df in calibration_data_run.groupby('run'):
                total_points = run_df['count'].sum()
                bin_weights = run_df['count'] / total_points
                ece = np.sum(
                    bin_weights * np.abs(run_df['fraction_of_positives'] - run_df['mean_predicted_prob'])
                )
                run_stats.append({'ece': ece})
            run_stats_df = pd.DataFrame(run_stats)

            ece, ece_std, ece_sem, ece_95ci_lower, ece_95ci_upper = get_statistics(run_stats_df['ece'])
            
            # Create dataframe with results, including 95% confidence intervals
            results_df = pd.DataFrame({
                'dataset_name': [dataset_name],
                'target': [target],
                'model': [model],
                'acq': [acq],
                'ece': [ece],
                'ece_std': [ece_std],
                'ece_sem': [ece_sem],
                'ece_95ci_lower': [ece_95ci_lower],
                'ece_95ci_upper': [ece_95ci_upper],
            })
            all_results.append(results_df)


        if all_results:
            pd.concat(all_results, ignore_index=True).sort_values(['dataset_name', 'model', 'acq']).to_csv(f'{save_dir}/calibration_results_iter{check_iter}.csv', index=False)
            


        # ---------- REGRESSION ----------

        # for regression and GP
        fnames = glob.glob(f'results/{datafile_name}/dat_files/*_gnn_*/results_mse*.pkl')
        fnames += glob.glob(f'results/{datafile_name}/dat_files/*_bnn_*/results_mse*.pkl')
        fnames += glob.glob(f'results/{datafile_name}/dat_files/*_gp_*/results_mse*.pkl')


        regression_results = []

        for f in tqdm(fnames, desc=f"Processing regression calibration files for {datafile_name}, iter {check_iter}"):
            res = pickle.load(open(f, "rb"))

            save_dir = f.split('/dat_files')[0] + '/calibration_regression'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)

            plot_df = []
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

                # Save correlation statistics for this run
                y_true = tmp_df['y_true'].to_numpy()
                y_pred = tmp_df['y_pred'].to_numpy()

                calib_df = regression_calibration_curve(y_pred, tmp_df['y_std'].to_numpy(), y_true, n_bins=10)
                calib_df["run"] = i
                plot_df.append(calib_df)
            
            if not plot_df:
                continue

            plot_df = pd.concat(plot_df)
            corr_stats = pd.DataFrame(corr_stats)
            
            # info from filename
            parts = f.split('/')[-2].split('_')
            dataset_name = parts[0]
            target = parts[1]
            model = parts[-2]
            acq = parts[-1]

            # gather by bins
            calibration_data = plot_df.groupby('bin').agg(
                mean_quantiles=('quantile', 'mean'),
                mean_cqs=('cq', 'mean'),
                sem_cqs=('cq', 'sem')
            ).dropna()

            # Plotting
            fig = plt.figure(figsize=(8, 8))
            plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
            
            plt.plot(calibration_data['mean_quantiles'], calibration_data['mean_cqs'], 's-', label='Model calibration')
            plt.fill_between(
                calibration_data['mean_quantiles'],
                calibration_data['mean_cqs'] - 1.96*calibration_data['sem_cqs'],
                calibration_data['mean_cqs'] + 1.96*calibration_data['sem_cqs'],
                alpha=0.2,
            )
            plt.title('C(q) Regression Calibration Plot')
            plt.xlabel('Quantile (q)')
            plt.ylabel('C(q) score')
            plt.xlim([0,1])
            plt.ylim([0,1])
            plt.legend()
            plt.grid(True, alpha=0.5)
            plt.axis('square')
            fig.savefig(f'{save_dir}/calibration-curve_{dataset_name}_{target}_{model}_{acq}_iter{check_iter}.png')
            plt.close()

            calibration_data_run = (
                plot_df.groupby(['run', 'bin']).agg(
                    mean_quantiles=('quantile', 'mean'),
                    mean_cqs=('cq', 'mean'),
                ).dropna().reset_index()  # This makes 'run' and 'bin' columns instead of indices
            )

            # Calculate statistics for each run in calibration_data_run, then aggregate
            run_stats = []
            for run, run_df in calibration_data_run.groupby('run'):
                # Calculate rECE (regression Expected Calibration Error) for this run
                rece = np.trapz(np.abs(run_df["mean_cqs"] - run_df["mean_quantiles"]), run_df["mean_quantiles"])
                run_stats.append({'ece': rece,})
            run_stats_df = pd.DataFrame(run_stats)

            # Aggregate statistics across runs (mean, std, sem, and 95% CI for ece)
            ece, ece_std, ece_sem, ece_95ci_lower, ece_95ci_upper = get_statistics(run_stats_df['ece'])
            
            # Create dataframe with results
            results_df = pd.DataFrame({
                'dataset_name': [dataset_name],
                'target': [target],
                'model': [model],
                'acq': [acq],
                'ece': [ece],
                'ece_std': [ece_std],
                'ece_sem': [ece_sem],
                'ece_95ci_lower': [ece_95ci_lower],
                'ece_95ci_upper': [ece_95ci_upper],
            })
            regression_results.append(results_df)

        if regression_results:
            pd.concat(regression_results, ignore_index=True).sort_values(['dataset_name', 'model', 'acq']).to_csv(f'{save_dir}/calibration_results_iter{check_iter}.csv', index=False)
