import sys
WORK_DIR = '../../..'
sys.path.append(WORK_DIR)

# from rogi import RoughnessIndex
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

# plotting parameters
sns.set_theme(style="white", font_scale=1.5)
sns.set_palette("colorblind")
plt.rc('lines', linewidth=2.0, markersize=8)

from sklearn.metrics import r2_score, ndcg_score
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr, pearsonr, kendalltau

from rbbo.metrics import frac_top_n, top_one, auc_metric
from rbbo.utils import ORACLE_NAMES, ORACLE_OBJ, ORACLE_ROGI, read_hparam_files

def is_overlap(mean1, ci1, mean2, ci2):
    high1 = mean1 + ci1
    low1 = mean1 - ci1
    high2 = mean2 + ci2
    low2 = mean2 - ci2

    top_overlap = (high1 > low2) & (high1 < high2)
    bottom_overlap = (high2 > low1) & (high2 < high1)

    return top_overlap + bottom_overlap

def get_ndcg(y_true, y_pred, goal, top_n=None):
    relevance_scores = MinMaxScaler().fit_transform(y_true)
    if goal == 'minimize':
        relevance_scores = -relevance_scores + 1.0
        y_pred = -y_pred
    score = ndcg_score(relevance_scores.T, y_pred.T, k=top_n)
    return score

if __name__ == "__main__":
    # input parameters
    top_n = 100
    iter_check = 19     # we check this evaluation for results (skips the num_init_design)

    all_results = []        # when we want to look across models/acq functions (per dataset)
    pred_results = []       
    cross_results = []

    for model_type in ['gnn','mlp','bnn']:
        for acq_func in ['ucb', 'greedy', 'ei']:

            plot_df = []        # this is for plotting over model/acq
            plot_pred = []

            for dataset_name in ORACLE_NAMES:
                goal = ORACLE_OBJ[dataset_name]
                rogi_score = ORACLE_ROGI[dataset_name]
                dataset_name = 'zinc_' + dataset_name
                dataset_path = f'{WORK_DIR}/data/{dataset_name}.csv'
                dataset: pd.DataFrame = pd.read_csv(dataset_path)

                rough_df = []           # this is for calculating individual statistics per dataset
                pred_collect = []       # this is for collecting prediction results for calculation

                for loss in ["mse", "ranking"]:
                    result_path = f'dat_files/{dataset_name}_{goal}_{model_type}_{acq_func}/results_{loss}_{dataset_name}_{goal}.pkl'
                    # result_path = f'{dataset_name}_{goal}_{model_type}_{acq_func}/results_{loss}_{dataset_name}_{goal}.pkl'
                    try:
                        results = pd.read_pickle(result_path)
                        hparams = read_hparam_files(f'dat_files/{dataset_name}_{goal}_{model_type}_{acq_func}/hparams.txt')
                    except:
                        continue

                    num_init = int(hparams['num_init_design'])
                    batch_size = int(hparams['batch_size'])

                    for i, res in enumerate(results):
                        if type(res) is tuple:
                            df, preds = res

                            gdf = preds[preds['iteration'] == iter_check]

                            r2 = r2_score(gdf['y_true'].tolist(), gdf['y_pred'].tolist())
                            ndcg = get_ndcg(gdf[['y_true']].to_numpy(), gdf[['y_pred']].to_numpy(), goal, top_n)
                            p_rho, _ = pearsonr(gdf['y_true'].tolist(), gdf['y_pred'].tolist())
                            s_rho, _ = spearmanr(gdf['y_true'].tolist(), gdf['y_pred'].tolist())
                            tau, _ = kendalltau(gdf['y_true'].tolist(), gdf['y_pred'].tolist())

                        else:
                            df = res
                        df['run'] = i
                        df['loss_type'] = loss
                        df = frac_top_n(dataset, df, top_n, goal)
                        df = top_one(df, goal)

                        auc = auc_metric(
                            dataset, 
                            bo_output = df, 
                            metric = "frac_top_n", 
                            goal = goal
                        )

                        rough_df.append({
                            "dataset": dataset_name,
                            "loss_type": loss,
                            "auc": auc,
                            "rogi": rogi_score,
                            "model_type": model_type,
                            "acq_func": acq_func,
                            "max_frac": df["frac_top_n"].max()
                        })

                        pred_collect.append({
                            'model_type': model_type,
                            'iteration': iter_check, 
                            'loss_type': loss,
                            'acq_func': acq_func,
                            'r2': r2, 
                            'ndcg': ndcg,
                            'spearman': s_rho,
                            'pearson': p_rho,
                            'kendall': tau,
                            'run': i,
                            'max_frac': df[df['evaluation'] == batch_size*iter_check + num_init]['frac_top_n'].values[0],
                            "dataset": dataset_name,
                            "rogi": rogi_score
                        })
                

                loss = "mse"
                result_path = f'dat_files/{dataset_name}_{goal}_gp_{acq_func}/results_{loss}_{dataset_name}_{goal}.pkl'
                # result_path = f'{dataset_name}_{goal}_gp_{acq_func}/results_{loss}_{dataset_name}_{goal}.pkl'

                results = pd.read_pickle(result_path)

                for i, res in enumerate(results):
                    if type(res) is tuple:
                        df, preds = res
                        gdf  = preds[preds['iteration'] == iter_check]

                        r2 = r2_score(gdf['y_true'].tolist(), gdf['y_pred'].tolist())
                        ndcg = get_ndcg(gdf[['y_true']].to_numpy(), gdf[['y_pred']].to_numpy(), goal, top_n)
                        p_rho, _ = pearsonr(gdf['y_true'].tolist(), gdf['y_pred'].tolist())
                        s_rho, _ = spearmanr(gdf['y_true'].tolist(), gdf['y_pred'].tolist())
                        tau, _ = kendalltau(gdf['y_true'].tolist(), gdf['y_pred'].tolist())

                    else:
                        df = res
                    df['run'] = i
                    df['loss_type'] = loss
                    df = frac_top_n(dataset, df, top_n, goal)
                    df = top_one(df, goal)

                    auc = auc_metric(
                        dataset, 
                        bo_output = df, 
                        metric = "frac_top_n", 
                        goal = goal
                    )
                    
                    # gather the optimization statistic
                    rough_df.append({
                        "dataset": dataset_name,
                        "loss_type": "gp + mll",
                        "auc": auc,
                        "rogi": rogi_score,
                        "model_type": "gp",
                        "acq_func": acq_func,
                        "max_frac": df["frac_top_n"].max()
                    })

                    pred_collect.append({
                        'model_type': "gp",
                        'iteration': iter_check, 
                        'loss_type': "gp + mll",
                        'acq_func': acq_func,
                        'r2': r2, 
                        'ndcg': ndcg,
                        'spearman': s_rho,
                        'pearson': p_rho,
                        'kendall': tau,
                        'run': i,
                        'max_frac': df[df['evaluation'] == batch_size*iter_check + num_init]['frac_top_n'].values[0],
                        "dataset": dataset_name,
                        "rogi": rogi_score
                    })

                rough_df = pd.DataFrame(rough_df)
                plot_df.append(rough_df)

                # get statistics before appending
                df_stats = []
                statistics = ["auc", "max_frac"]
                for _, gdf in rough_df.groupby("loss_type"):
                    for stats in statistics:
                        gdf[f'{stats}_mean'] = gdf[stats].mean()
                        gdf[f'{stats}_std'] = gdf[stats].std()
                        gdf[f'{stats}_95ci'] = gdf[f'{stats}_std'] / np.sqrt(len(gdf)) * 1.96
                        gdf = gdf.drop(stats, axis=1)
                    df_stats.append(gdf.iloc[0])
                df_stats = pd.DataFrame(df_stats)
                all_results.append(df_stats)

                # calculate average and 
                pred_collect = pd.DataFrame(pred_collect)
                df_stats = []
                for _, gdf in pred_collect.groupby("loss_type"):
                    for stats in ['max_frac', 'spearman', 'r2', 'kendall', 'ndcg']:
                        gdf[f'{stats}_mean'] = gdf[stats].mean()
                        gdf[f'{stats}_std'] = gdf[stats].std()
                        gdf[f'{stats}_95ci'] = gdf[f'{stats}_std'] / np.sqrt(len(gdf)) * 1.96
                        gdf = gdf.drop(stats, axis=1)     
                    df_stats.append(gdf.iloc[0])
                df_stats = pd.DataFrame(df_stats)
                plot_pred.append(df_stats)

            # plot rogi for sets of models/acq
            plot_df = pd.concat(plot_df)
            plot_df = plot_df.replace('_', ' ', regex=True)
            plot_df = plot_df.replace('zinc ', '', regex=True)
            plot_pred = pd.concat(plot_pred)

            pred_results.append(plot_pred)

            ##### PLOT statistic per dataset (for each model/acq)
            ax = sns.lineplot(plot_df, x="rogi", y="auc", hue="loss_type", marker='o', linestyle='', err_style='bars', hue_order=['mse', 'ranking', 'gp + mll'], alpha=0.95)
            # sns.lineplot(plot_df, x="rogi", y="auc", hue="loss_type", marker='o', linestyle='', err_style='bars', hue_order=['mse', 'ranking'], alpha=0.95)
            ax.get_legend().set_title("")
            plt.xlabel('Dataset roughness (ROGI)')
            plt.ylabel('BO performance (AUC)')
            # plt.ylim([0, 0.25])        # will need to adjust based on your runs
            plt.savefig(f'rogi_auc_{model_type}_{acq_func}.png', bbox_inches='tight')
            plt.close()
            
            # 
            sns.lineplot(plot_df, x="rogi", y="max_frac", hue="loss_type", marker='o', linestyle='', err_style='bars', hue_order=['mse', 'ranking', 'gp + mll'], alpha=0.95)
            # sns.lineplot(plot_df, x="rogi", y="auc", hue="loss_type", marker='o', linestyle='', err_style='bars', hue_order=['mse', 'ranking'], alpha=0.95)
            plt.xlabel('ROGI')
            plt.ylabel('Maximum fraction of top 100 found')
            # plt.ylim([0, 0.25])        # will need to adjust based on your runs
            plt.savefig(f'rogi_max_frac_{model_type}_{acq_func}.png', bbox_inches='tight')
            plt.close()


            fig, ax  = plt.subplots(figsize=(12, 6))
            sns.barplot(plot_df, x="dataset", y="auc", hue="loss_type", hue_order=['mse', 'ranking', 'gp + mll'], order=plot_df.sort_values('rogi')["dataset"] ,alpha=0.95, ax=ax)
            ax.tick_params(axis='x', rotation=90)
            ax.set_xlabel(None)
            ax.set_ylabel('AUC')
            ax.set_ylim([0, 0.35])         # will need to adjust based on your runs
            fig.savefig(f'auc_{model_type}_{acq_func}_bar.png', bbox_inches='tight')
            plt.close()

            fig, ax  = plt.subplots(figsize=(12, 6))
            sns.barplot(plot_df, x="dataset", y="max_frac", hue="loss_type", hue_order=['mse', 'ranking', 'gp + mll'], order=plot_df.sort_values('rogi')["dataset"], alpha=0.95, ax=ax)
            ax.tick_params(axis='x', rotation=90)
            ax.set_xlabel(None)
            ax.set_ylabel('Fraction of top 100 found')
            ax.set_ylim([0, 0.5])        # will need to adjust based on your runs
            fig.savefig(f'max_frac_{model_type}_{acq_func}_bar.png', bbox_inches='tight')
            plt.close()

            ##### PLOT scatter of surrogate stat vs. some stat (for each model/acq)
            palette = sns.color_palette('colorblind')
            for corr in ['spearman', 'kendall', 'r2', 'ndcg']:
                ax = sns.scatterplot(plot_pred, x="max_frac_mean", y=f"{corr}_mean", hue="loss_type", marker='o', hue_order=['mse', 'ranking', 'gp + mll'], alpha=0.95)
                ax.get_legend().set_title("")
                for k, loss_type in enumerate(['mse', 'ranking', 'gp + mll']):
                    subdf = plot_pred[plot_pred['loss_type'] == loss_type].sort_values("max_frac_mean")
                    try:
                        plt.errorbar(subdf["max_frac_mean"], subdf[f"{corr}_mean"], 
                                xerr=subdf["max_frac_95ci"], yerr=subdf[f"{corr}_95ci"],
                                fmt='none', alpha=0.3, color=palette[k])
                        # make a linear fit
                        m,b = np.polyfit(subdf["max_frac_mean"], subdf[f"{corr}_mean"], 1)
                        lin_fit=m*subdf["max_frac_mean"]+b
                        plt.plot(subdf["max_frac_mean"], lin_fit, color=palette[k], alpha=0.7)
                        plt.ylim([-0.1, 1])
                        p_rho, p_p = pearsonr(subdf["max_frac_mean"], subdf[f"{corr}_mean"])
                        cross_results.append({
                            'model_type': "gp" if loss_type == "gp + mll" else model_type,
                            'loss_type': loss_type,
                            'acq_func': acq_func,
                            'pearson': p_rho,
                            'pearson_p': p_p,
                            'slope': m,
                            'intercept': b,
                            'correlation': corr,
                        })
                    except:
                        # not all models correspond to all losses/acq
                        pass
                # plt.legend()
                plt.xlabel('Fraction found')
                plt.ylabel(f'{corr.capitalize()}')
                # plt.ylim([0, 0.25])        # will need to adjust based on your runs
                plt.savefig(f'perf_v_opt_{corr}_{model_type}_{acq_func}.png', bbox_inches='tight')
                plt.close()

                # plot the surrogate v rogi
                ax = sns.scatterplot(plot_pred, x="rogi", y=f"{corr}_mean", hue="loss_type", marker='o', hue_order=['mse', 'ranking', 'gp + mll'], alpha=0.95)
                ax.get_legend().set_title("")
                for k, loss_type in enumerate(['mse', 'ranking', 'gp + mll']):
                    subdf = plot_pred[plot_pred['loss_type'] == loss_type].sort_values("rogi")
                    try:
                        plt.errorbar(subdf["rogi"], subdf[f"{corr}_mean"], 
                                yerr=subdf[f"{corr}_95ci"], fmt='none', alpha=0.3, color=palette[k])
                        # make a linear fit
                        m,b = np.polyfit(subdf["rogi"], subdf[f"{corr}_mean"], 1)
                        lin_fit=m*subdf["rogi"]+b
                        plt.plot(subdf["rogi"], lin_fit, color=palette[k], alpha=0.7)
                        plt.ylim([-0.1, 1])
                        # cross_results.append({
                        #     'model_type': "gp" if loss_type == "gp + mll" else model_type,
                        #     'loss_type': loss_type,
                        #     'acq_func': acq_func,
                        #     'pearson': pearsonr(subdf["rogi"], subdf[f"{corr}_mean"])[0],
                        #     'slope': m,
                        #     'intercept': b,
                        #     'correlation': corr,
                        # })
                    except:
                        # not all models correspond to all losses/acq
                        pass
                # plt.legend()
                plt.xlabel('ROGI')
                plt.ylabel(f'{corr.capitalize()}')
                # plt.ylim([0, 0.25])        # will need to adjust based on your runs
                plt.savefig(f'perf_v_rogi_{corr}_{model_type}_{acq_func}.png', bbox_inches='tight')
                plt.close()


    ##### plot the AVERAGE statistic each individual dataset
    pred_results = pd.concat(pred_results)
    pred_results = pred_results[~(pred_results['acq_func'] == 'ei')]        # get rid of ei
    pred_results = pred_results[~((pred_results['acq_func'] == 'greedy') * (pred_results['model_type'] == 'gnn'))] 
    pred_results = pred_results[~((pred_results['acq_func'] == 'greedy') * (pred_results['model_type'] == 'bnn'))] 
    pred_results = pred_results[~((pred_results['acq_func'] == 'greedy') * (pred_results['model_type'] == 'gp'))] 
    ax = sns.violinplot(pred_results, x='model_type', y='kendall_mean', hue='loss_type', hue_order=['mse', 'ranking', 'gp + mll'], order=['bnn', 'gnn', 'mlp', 'gp'])
    ax.get_legend().set_title("")
    ax.set_ylim([-0.2, 0.8])
    ax.set_ylabel('Average test Kendall tau')
    ax.set_xlabel('')
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.savefig(f'_kendall_at_iter{iter_check}.png', bbox_inches='tight')
    plt.close()
        




    ##### LATEX table for cross dataset results
    import pdb; pdb.set_trace()
    cross_results = pd.DataFrame(cross_results)
    cross_results = cross_results.sort_values(['model_type', 'loss_type', 'acq_func'])
    cross_results = cross_results.drop_duplicates(['loss_type', 'acq_func', 'model_type', 'correlation']) # gps will be duplicated for plots
    cross_results = cross_results[cross_results['correlation'] != 'spearman']
    cross_results = cross_results[cross_results['correlation'] != 'ndcg']

    for mt in ['gnn','mlp','bnn','gp']:
        for af in ['ei', 'greedy', 'ucb']:
            # select the model and acq
            tmp = cross_results[cross_results['acq_func'] == af]
            tmp = tmp[tmp['model_type'] == mt]
            tmp = tmp.drop(['model_type', 'acq_func'], axis=1)

            if tmp.empty:
                continue

            cols = ["pearson", "slope", "intercept"]
            tmp = pd.pivot(tmp, values=cols, index=['correlation'], columns=['loss_type'])

            # tmp = pd.pivot_table(tmp, values=['auc'], index=['dataset'], aggfunc='first')
            # tmp = pd.pivot(tmp, values=statistics, index=['dataset'], columns=['loss'])
            tmp = tmp.rename(index=lambda name: name.capitalize())
            tmp.to_latex(f'latex_table_{af}_{mt}_perf_v_opt.txt', float_format="{:.3f}".format, index_names=False)


    ##### GATHER RESULTS FOR LATEX TABLE
    all_results = pd.concat(all_results)
    all_results = all_results.sort_values(['rogi', 'model_type', 'loss_type', 'acq_func'])
    all_results = all_results.drop_duplicates(['dataset', 'loss_type', 'acq_func', 'model_type']) # gps will be duplicated for plots
    all_results = all_results.rename(columns={'loss_type': 'loss'})
    all_results = all_results.replace('_', ' ', regex=True)
    all_results = all_results.replace('zinc ', '', regex=True)
        

    for mt in ['gnn','mlp','bnn','gp']:
        for af in ['ei', 'greedy', 'ucb']:
            # select the model and acq
            tmp = all_results[all_results['acq_func'] == af]
            tmp = tmp[tmp['model_type'] == mt]
            tmp = tmp.drop(['model_type', 'acq_func'], axis=1)

            if tmp.empty:
                continue

            cols = [f'{stats}_mean' for stats in statistics]
            cols += [f'{stats}_95ci' for stats in statistics]
            tmp = pd.pivot(tmp, values=cols, index=['dataset'], columns=['loss'])

            keep_cols = []
            for stats in statistics:
                if mt != 'gp':
                    tmp[f'{stats}_greater'] = tmp[f'{stats}_mean']['mse'] < tmp[f'{stats}_mean']['ranking']
                    tmp[f'{stats}_overlap'] = is_overlap(tmp[f'{stats}_mean']['mse'], tmp[f'{stats}_95ci']['mse'], tmp[f'{stats}_mean']['ranking'], tmp[f'{stats}_95ci']['ranking'])

                    tmp[f'{stats}_mse'] = tmp[f'{stats}_mean']['mse'].apply(lambda x:f'{x:.4f}') + ' ± ' + tmp[f'{stats}_95ci']['mse'].apply(lambda x:f'{x:.4f}')
                    tmp[f'{stats}_ranking'] = tmp[f'{stats}_mean']['ranking'].apply(lambda x:f'{x:.4f}') + ' ± ' + tmp[f'{stats}_95ci']['ranking'].apply(lambda x:f'{x:.4f}')

                    # bold for better and no overlap
                    tmp.loc[tmp[f'{stats}_greater']*~tmp[f'{stats}_overlap'], f'{stats}_ranking']= '\\bftab{' + tmp[tmp[f'{stats}_greater']*~tmp[f'{stats}_overlap']][f'{stats}_ranking'] + '}'
                    tmp.loc[~tmp[f'{stats}_greater']*~tmp[f'{stats}_overlap'], f'{stats}_mse']= '\\bftab{' + tmp[~tmp[f'{stats}_greater']*~tmp[f'{stats}_overlap']][f'{stats}_mse'] + '}'

                    # underline for better but overlap
                    tmp.loc[tmp[f'{stats}_greater']*tmp[f'{stats}_overlap'], f'{stats}_ranking']= '\\underline{' + tmp[tmp[f'{stats}_greater']*tmp[f'{stats}_overlap']][f'{stats}_ranking'] + '}'
                    tmp.loc[~tmp[f'{stats}_greater']*tmp[f'{stats}_overlap'], f'{stats}_mse']= '\\underline{' + tmp[~tmp[f'{stats}_greater']*tmp[f'{stats}_overlap']][f'{stats}_mse'] + '}'

                    keep_cols += [f'{stats}_mse', f'{stats}_ranking']
                else:
                    tmp[stats] = tmp[f'{stats}_mean']['gp + mll'].apply(lambda x:f'{x:.4f}') + ' ± ' + tmp[f'{stats}_95ci']['gp + mll'].apply(lambda x:f'{x:.4f}')
                    keep_cols += [stats]

            tmp = tmp[keep_cols]

            # tmp = pd.pivot_table(tmp, values=['auc'], index=['dataset'], aggfunc='first')
            # tmp = pd.pivot(tmp, values=statistics, index=['dataset'], columns=['loss'])
            tmp = tmp.rename(columns=lambda name: name.replace('_', ' '))
            tmp.to_latex(f'latex_table_{af}_{mt}.txt', float_format="{:.4f}".format, index_names=False)
