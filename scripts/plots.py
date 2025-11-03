data_dir = 'results'

import os
import pandas as pd

datanames = ["ROGI", "KI", "EC50"]
calib_types = ["calibration_regression", "calibration_ranking"]

results = {}

import matplotlib.pyplot as plt
import seaborn as sns

# Generate LaTeX tables for each dataname and calib_type, for each check_iter
for check_iter in [0, 49, 99]:
    for dataname in datanames:
        for calib_type in calib_types:
            folder = os.path.join(data_dir, dataname, calib_type)
            calib_csv_path = os.path.join(folder, f"calibration_results_iter{check_iter}.csv")
            perf_csv_path = os.path.join(folder, f"performance_results_iter{check_iter}.csv")
            if not (os.path.exists(calib_csv_path) and os.path.exists(perf_csv_path)):
                continue

            calib_df = pd.read_csv(calib_csv_path)
            perf_df = pd.read_csv(perf_csv_path)

            # Append dataset_name to target for both dataframes
            if 'target' in calib_df.columns:
                calib_df['dataset_name'] = calib_df['dataset_name'].astype(str) + " " + calib_df['target'].astype(str)
            if 'target' in perf_df.columns:
                perf_df['dataset_name'] = perf_df['dataset_name'].astype(str) + " " + perf_df['target'].astype(str)

            # Prepare a list of unique dataset names
            dataset_names = sorted(set(calib_df['dataset_name']))

            # Prepare LaTeX table header with multirow for BNN + UCB and GNN + UCB, and grouping ECE and tau
            latex = []
            latex.append("\\begin{table*}[ht]")
            latex.append("\\centering")
            latex.append(f"\\caption{{{dataname} - {calib_type.replace('calibration_', '').capitalize()} models (iteration {check_iter})}}")
            # latex.append("\\setlength\\tabcolsep{4pt}")

            # Set ECE label depending on calibration type
            if calib_type == "calibration_regression":
                ece_label = "rECE"
            else:
                ece_label = "prECE"

            # If regression, add GP + UCB columns
            if calib_type == "calibration_regression":
                latex.append("\\begin{tabular}{lcc|cc|cc|c}")
                latex.append("\\toprule")
                # First header row: group BNN + UCB, GNN + UCB, GP + UCB columns
                latex.append(" & \\multicolumn{2}{c}{{BNN + UCB}} & \\multicolumn{2}{c}{{GNN + UCB}} & \\multicolumn{2}{c}{{GP + UCB}} & {MLP + greedy} \\\\")
                latex.append("\cmidrule(lr){2-3}\cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-8}")
                # Second header row: just the statistic names, with correct ECE label
                latex.append(f"Dataset & {ece_label} & $\\tau$ & {ece_label} & $\\tau$ & {ece_label} & $\\tau$ & $\\tau$ \\\\")
            else:
                latex.append("\\begin{tabular}{lcc|cc|c}")
                latex.append("\\toprule")
                # First header row: group BNN + UCB and GNN + UCB columns
                latex.append(" & \\multicolumn{2}{c}{{BNN + UCB}} & \\multicolumn{2}{c}{{GNN + UCB}} & {MLP + greedy} \\\\")
                latex.append("\cmidrule(lr){2-3}\cmidrule(lr){4-5} \cmidrule(lr){6-6}")
                # Second header row: just the statistic names, with correct ECE label
                latex.append(f"Dataset & {ece_label} & $\\tau$ & {ece_label} & $\\tau$ & $\\tau$ \\\\")

            latex.append("\\midrule")

            collect_metrics = {}
            for ds in dataset_names:
                # ECE and 95% CI for BNN + UCB, GNN + UCB, and (if regression) GP + UCB
                def ece_ci_pm(df, model, acq):
                    row = df[(df['dataset_name'] == ds) & (df['model'] == model) & (df['acq'] == acq)]
                    if row.shape[0] == 0:
                        return "", None
                    ece = row.iloc[0]['ece']
                    ece_95ci = row.iloc[0]['ece_sem'] * 1.96
                    return f"{ece:.3f} $\\pm$ {ece_95ci:.3f}", ece
                    

                bnn_ece, bnn_ece_val = ece_ci_pm(calib_df, 'bnn', 'ucb')
                gnn_ece, gnn_ece_val = ece_ci_pm(calib_df, 'gnn', 'ucb')
                collect_metrics[ds+'_ece'] = {'bnn': bnn_ece_val, 'gnn': gnn_ece_val}
                if calib_type == "calibration_regression":
                    gp_ece, gp_ece_val = ece_ci_pm(calib_df, 'gp', 'ucb')
                    collect_metrics[ds+'_ece'].update({'gp':gp_ece_val})

                # Kendall tau and 95% CI for BNN + UCB, GNN + UCB, (if regression) GP + UCB, and MLP + greedy from perf_df
                def kendall_ci_pm(df, model, acq):
                    row = df[(df['dataset_name'] == ds) & (df['model'] == model) & (df['acq'] == acq)]
                    val = row.iloc[0]['kendall']
                    val_95ci = row.iloc[0]['kendall_sem'] * 1.96
                    return f"{val:.3f} $\\pm$ {val_95ci:.3f}", val

                bnn_tau, bnn_tau_val = kendall_ci_pm(perf_df, 'bnn', 'ucb')
                gnn_tau, gnn_tau_val = kendall_ci_pm(perf_df, 'gnn', 'ucb')
                mlp_greedy_tau, mlp_greedy_tau_val = kendall_ci_pm(perf_df, 'mlp', 'greedy')
                collect_metrics[ds+'_tau'] = {'bnn': bnn_tau_val, 'gnn': gnn_tau_val, 'mlp': mlp_greedy_tau_val}
                if calib_type == "calibration_regression":
                    gp_tau, gp_tau_val = kendall_ci_pm(perf_df, 'gp', 'ucb')
                    collect_metrics[ds+'_tau'].update({'gp': gp_tau_val})

                if calib_type == "calibration_regression":
                    latex.append(f"{ds} & {bnn_ece} & {bnn_tau} & {gnn_ece} & {gnn_tau} & {gp_ece} & {gp_tau} & {mlp_greedy_tau} \\\\")
                else:
                    latex.append(f"{ds} & {bnn_ece} & {bnn_tau} & {gnn_ece} & {gnn_tau} & {mlp_greedy_tau} \\\\")

            latex.append("\\bottomrule")
            latex.append("\\end{tabular}")
            latex.append("\\end{table*}")
            latex_str = "\n".join(latex)

            # Save LaTeX table to file with iteration suffix
            out_name = f"{dataname}_{calib_type.replace('calibration_', '')}_summary_table_iter{check_iter}.tex"
            out_path = os.path.join(data_dir, out_name)
            with open(out_path, "w") as f:
                f.write(latex_str)

            # This has the metrics for all the datasets for every model
            collect_metrics = pd.DataFrame(collect_metrics).transpose()


# This is for the average across dataset type ##################################
for dataname in datanames:
    for calib_type in calib_types:
        for check_iter in [0, 49, 99]:
            folder = os.path.join(data_dir, dataname, calib_type)
            calib_csv_path = os.path.join(folder, f"calibration_results_iter{check_iter}.csv")
            perf_csv_path = os.path.join(folder, f"performance_results_iter{check_iter}.csv")
            if not (os.path.exists(calib_csv_path) and os.path.exists(perf_csv_path)):
                continue

            calib_df = pd.read_csv(calib_csv_path)
            perf_df = pd.read_csv(perf_csv_path)

            # Append dataset_name to target for both dataframes
            if 'target' in calib_df.columns:
                calib_df['dataset_name'] = calib_df['dataset_name'].astype(str) + " " + calib_df['target'].astype(str)
            if 'target' in perf_df.columns:
                perf_df['dataset_name'] = perf_df['dataset_name'].astype(str) + " " + perf_df['target'].astype(str)

            # For each model/acq, collect the ECE and tau values across datasets
            stats = {}

            # Define which models/acqs to include
            if calib_type == "calibration_regression":
                model_acqs = [
                    ("bnn", "ucb"),
                    ("gnn", "ucb"),
                    ("gp", "ucb"),
                    ("mlp", "greedy"),
                ]
            else:
                model_acqs = [
                    ("bnn", "ucb"),
                    ("gnn", "ucb"),
                    ("mlp", "greedy"),
                ]

            # For each model/acq, collect ECE and tau
            for model, acq in model_acqs:
                # ECE
                if model != "mlp":
                    ece_vals = []
                    for ds in sorted(set(calib_df['dataset_name'])):
                        row = calib_df[(calib_df['dataset_name'] == ds) & (calib_df['model'] == model) & (calib_df['acq'] == acq)]
                        if row.shape[0] > 0 and 'ece' in row.columns:
                            val = row.iloc[0]['ece']
                            if not pd.isna(val):
                                ece_vals.append(val)
                    stats[(model, acq, "ece")] = ece_vals
                # tau
                tau_vals = []
                for ds in sorted(set(perf_df['dataset_name'])):
                    row = perf_df[(perf_df['dataset_name'] == ds) & (perf_df['model'] == model) & (perf_df['acq'] == acq)]
                    if row.shape[0] > 0 and 'kendall' in row.columns:
                        val = row.iloc[0]['kendall']
                        if not pd.isna(val):
                            tau_vals.append(val)
                stats[(model, acq, "tau")] = tau_vals

            # Prepare LaTeX table
            latex = []
            latex.append("\\begin{table*}[ht]")
            latex.append("\\centering")
            latex.append("\\caption{Average ECE and $\\tau$ for each dataname}")
            if calib_type == "calibration_regression":
                latex.append("\\begin{tabular}{lcc|cc|cc|c}")
                latex.append("\\toprule")
                latex.append(" & \\multicolumn{2}{c}{{BNN + UCB}} & \\multicolumn{2}{c}{{GNN + UCB}} & \\multicolumn{2}{c}{{GP + UCB}} & {MLP + greedy} \\\\")
                latex.append("\cmidrule(lr){2-3}\cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-8}")
                if calib_type == "calibration_regression":
                    ece_label = "rECE"
                else:
                    ece_label = "prECE"
                latex.append(f"Dataname & {ece_label} & $\\tau$ & {ece_label} & $\\tau$ & {ece_label} & $\\tau$ & $\\tau$ \\\\")
            else:
                latex.append("\\begin{tabular}{lcc|cc|c}")
                latex.append("\\toprule")
                latex.append(" & \\multicolumn{2}{c}{{BNN + UCB}} & \\multicolumn{2}{c}{{GNN + UCB}} & {MLP + greedy} \\\\")
                latex.append("\cmidrule(lr){2-3}\cmidrule(lr){4-5} \cmidrule(lr){6-6}")
                if calib_type == "calibration_regression":
                    ece_label = "rECE"
                else:
                    ece_label = "prECE"
                latex.append(f"Dataname & {ece_label} & $\\tau$ & {ece_label} & $\\tau$ & $\\tau$ \\\\")
            latex.append("\\midrule")

            # Helper to format mean and error as x.xx $^{{+err_hi}}_{{-err_lo}}$
            def mean_pm_err(vals):
                if len(vals) == 0:
                    return ""
                mean = float(pd.Series(vals).mean())
                ci95_lower = float(pd.Series(vals).quantile(0.025))
                ci95_upper = float(pd.Series(vals).quantile(0.975))
                return f"{mean:.3f} $^{{+{ci95_upper-mean:.3f}}}_{{-{mean-ci95_lower:.3f}}}$"

            # Compose the row for this dataname
            row = [dataname]
            if calib_type == "calibration_regression":
                # BNN + UCB
                row.append(mean_pm_err(stats.get(("bnn", "ucb", "ece"), [])))
                row.append(mean_pm_err(stats.get(("bnn", "ucb", "tau"), [])))
                # GNN + UCB
                row.append(mean_pm_err(stats.get(("gnn", "ucb", "ece"), [])))
                row.append(mean_pm_err(stats.get(("gnn", "ucb", "tau"), [])))
                # GP + UCB
                row.append(mean_pm_err(stats.get(("gp", "ucb", "ece"), [])))
                row.append(mean_pm_err(stats.get(("gp", "ucb", "tau"), [])))
                # MLP + greedy (only tau)
                row.append(mean_pm_err(stats.get(("mlp", "greedy", "tau"), [])))
            else:
                # BNN + UCB
                row.append(mean_pm_err(stats.get(("bnn", "ucb", "ece"), [])))
                row.append(mean_pm_err(stats.get(("bnn", "ucb", "tau"), [])))
                # GNN + UCB
                row.append(mean_pm_err(stats.get(("gnn", "ucb", "ece"), [])))
                row.append(mean_pm_err(stats.get(("gnn", "ucb", "tau"), [])))
                # MLP + greedy (only tau)
                row.append(mean_pm_err(stats.get(("mlp", "greedy", "tau"), [])))
            latex.append(" & ".join(row) + " \\\\")
            latex.append("\\hline")
            latex.append("\\end{tabular}")
            latex.append("\\end{table*}")
            latex_str = "\n".join(latex)

            # Save LaTeX table to file with iteration suffix
            out_name = f"{dataname}_{calib_type.replace('calibration_', '')}_summary_table_averages_iter{check_iter}.tex"
            out_path = os.path.join(data_dir, out_name)
            with open(out_path, "w") as f:
                f.write(latex_str)



# Generate LaTeX tables for each dataname and only regression but for R2 score
for check_iter in [0, 49, 99]:
    for dataname in datanames:
        for calib_type in ['calibration_regression']:
            folder = os.path.join(data_dir, dataname, calib_type)
            calib_csv_path = os.path.join(folder, f"calibration_results_iter{check_iter}.csv")
            perf_csv_path = os.path.join(folder, f"performance_results_iter{check_iter}.csv")
            if not (os.path.exists(calib_csv_path) and os.path.exists(perf_csv_path)):
                continue

            calib_df = pd.read_csv(calib_csv_path)
            perf_df = pd.read_csv(perf_csv_path)

            # Append dataset_name to target for both dataframes
            if 'target' in calib_df.columns:
                calib_df['dataset_name'] = calib_df['dataset_name'].astype(str) + " " + calib_df['target'].astype(str)
            if 'target' in perf_df.columns:
                perf_df['dataset_name'] = perf_df['dataset_name'].astype(str) + " " + perf_df['target'].astype(str)

            # Prepare a list of unique dataset names
            dataset_names = sorted(set(calib_df['dataset_name']))

            # Prepare LaTeX table header with multirow for BNN + UCB and GNN + UCB, and grouping ECE and tau
            latex = []
            latex.append("\\begin{table*}[ht]")
            latex.append("\\centering")
            latex.append(f"\\caption{{{dataname} - {calib_type.replace('calibration_', '').capitalize()} models (iteration {check_iter})}}")
            # latex.append("\\setlength\\tabcolsep{4pt}")

            latex.append("\\begin{tabular}{lc|c|c|c}")
            latex.append("\\toprule")
            # First header row: group BNN + UCB, GNN + UCB, GP + UCB columns
            latex.append(" & BNN + UCB & GNN + UCB & GP + UCB & MLP + greedy \\\\")

            latex.append("\\midrule")

            collect_metrics = {}
            for ds in dataset_names:
                # ECE and 95% CI for BNN + UCB, GNN + UCB, and (if regression) GP + UCB
                def ece_ci_pm(df, model, acq):
                    row = df[(df['dataset_name'] == ds) & (df['model'] == model) & (df['acq'] == acq)]
                    if row.shape[0] == 0:
                        return "", None
                    ece = row.iloc[0]['ece']
                    ece_95ci = row.iloc[0]['ece_sem'] * 1.96
                    return f"{ece:.3f} $\\pm$ {ece_95ci:.3f}", ece
                    
                bnn_ece, bnn_ece_val = ece_ci_pm(calib_df, 'bnn', 'ucb')
                gnn_ece, gnn_ece_val = ece_ci_pm(calib_df, 'gnn', 'ucb')
                collect_metrics[ds+'_ece'] = {'bnn': bnn_ece_val, 'gnn': gnn_ece_val}
                if calib_type == "calibration_regression":
                    gp_ece, gp_ece_val = ece_ci_pm(calib_df, 'gp', 'ucb')
                    collect_metrics[ds+'_ece'].update({'gp':gp_ece_val})

                # R2 and 95% CI for BNN + UCB, GNN + UCB, (if regression) GP + UCB, and MLP + greedy from perf_df
                def r2_ci_pm(df, model, acq):
                    row = df[(df['dataset_name'] == ds) & (df['model'] == model) & (df['acq'] == acq)]
                    val = row.iloc[0]['r2']
                    val_95ci = row.iloc[0]['r2_sem'] * 1.96
                    return f"{val:.3f} $\\pm$ {val_95ci:.3f}", val

                bnn_r2, bnn_r2_val = r2_ci_pm(perf_df, 'bnn', 'ucb')
                gnn_r2, gnn_r2_val = r2_ci_pm(perf_df, 'gnn', 'ucb')
                mlp_greedy_r2, mlp_greedy_r2_val = r2_ci_pm(perf_df, 'mlp', 'greedy')
                collect_metrics[ds+'_r2'] = {'bnn': bnn_r2_val, 'gnn': gnn_r2_val, 'mlp': mlp_greedy_r2_val}
                if calib_type == "calibration_regression":
                    gp_r2, gp_r2_val = r2_ci_pm(perf_df, 'gp', 'ucb')
                    collect_metrics[ds+'_r2'].update({'gp': gp_r2_val})

                if calib_type == "calibration_regression":
                    latex.append(f"{ds} & {bnn_r2} & {gnn_r2} & {gp_r2} & {mlp_greedy_r2} \\\\")
                else:
                    latex.append(f"{ds} & {bnn_r2} & {gnn_r2} & {mlp_greedy_r2} \\\\")

            latex.append("\\bottomrule")
            latex.append("\\end{tabular}")
            latex.append("\\end{table*}")
            latex_str = "\n".join(latex)

            # Save LaTeX table to file with iteration suffix
            out_name = f"{dataname}_{calib_type.replace('calibration_', '')}_R2_summary_table_iter{check_iter}.tex"
            out_path = os.path.join(data_dir, out_name)
            with open(out_path, "w") as f:
                f.write(latex_str)

            # This has the metrics for all the datasets for every model
            collect_metrics = pd.DataFrame(collect_metrics).transpose()

