import pandas as pd
import glob
import os, sys
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sys.path.append('../')


# plotting parameters
sns.set_theme(style="white", font_scale=1.5)
sns.set_palette("colorblind")
plt.rc("lines", linewidth=2.0, markersize=8)

loss_map = {'ranking': 'ranking', 'mse': 'regression'}
acq_map = {'ucb': 'UCB', 'ei': 'EI', 'greedy': 'greedy strategy'}


# git clone MoleculeACE repo into the path
data_path = '../../MoleculeACE/MoleculeACE/Data/benchmark_data/'

csv_files = glob.glob(os.path.join(data_path, "*.csv"))

dataframes = {}
for file in csv_files:
    df = pd.read_csv(file)
    dataset_name = os.path.splitext(file)[0].split('/')[-1]

    orig_df = pd.read_csv(f'../data/{dataset_name}.csv')
    orig_df.dropna(inplace=True)
    orig_df["cliff_mol"] = 0

    cliff_smiles = set(df.loc[df["cliff_mol"] == 1, "smiles"])
    orig_df.loc[orig_df["smiles"].isin(cliff_smiles), "cliff_mol"] = 1

    orig_df, _ = train_test_split(orig_df, test_size=0.15, random_state=42)   # this was the exact split used in the BO

    # Get indices of top 100 by target (lowest values)
    top100_indices = orig_df.nsmallest(100, 'target').index
    orig_df['top100'] = False
    orig_df.loc[top100_indices, 'top100'] = True
    dataframes[dataset_name] = orig_df

if not os.path.exists('cliff_analysis.csv'):
    collect = []
    for datafile_name in ['EC50', 'KI']:
        fnames = glob.glob(f'results/{datafile_name}/dat_files/*/results_*.pkl')

        for f in tqdm(fnames, desc=f"Processing ranking calibration files for {datafile_name}"):
            res = pickle.load(open(f, 'rb'))

            # Extract dataset name from the file path
            fname_base = os.path.basename(f)
            dataset_name = f.split('/dat_files/')[1].split('/')[0].split('_')[0] + '_' + f.split('/dat_files/')[1].split('/')[0].split('_')[1]

            # Extract model, acquisition function, and loss from the file path
            dir_name = os.path.basename(os.path.dirname(f))
            dir_parts = dir_name.split('_')
            if len(dir_parts) >= 5:
                model = dir_parts[-2]
                acquisition = dir_parts[-1]
            else:
                model = None
                acquisition = None

            # Extract loss function from the filename
            fname_base = os.path.basename(f)
            if fname_base.startswith("results_"):
                fname_parts = fname_base.split('_')
                if len(fname_parts) >= 3:
                    loss = fname_parts[1]
                else:
                    loss = None
            else:
                loss = None

            cliff_df = dataframes[dataset_name]

            # Collect results for each dataset_name in a list of dicts, then save as a DataFrame
            results = []
            for i in range(len(res)):
                df = res[i][0]
                # Find which smiles in df["smiles"] are in cliff_df, and count how many are in top100 or cliff
                smiles_in_df = set(df["smiles"])
                # Subset cliff_df to those with smiles in df
                matched = cliff_df[cliff_df["smiles"].isin(smiles_in_df)]
                # Count how many are top100
                n_top100 = matched["top100"].sum() if "top100" in matched.columns else 0
                # Count how many are cliff (if exists)
                n_cliff = matched["cliff_mol"].sum() if "cliff_mol" in matched.columns else 0

                # For the original dataset (cliff_df), count number of top100 True and n_cliff == 1
                n_top100_orig = cliff_df["top100"].sum() if "top100" in cliff_df.columns else 0
                n_cliff_orig = cliff_df["cliff_mol"].sum() if "cliff_mol" in cliff_df.columns else 0

                # Count how many are both top100 and cliff
                if "top100" in matched.columns and "cliff_mol" in matched.columns:
                    n_both = ((matched["top100"]) & (matched["cliff_mol"])).sum()
                else:
                    n_both = 0
                    
                results.append({
                    "loss": loss,
                    "model": model,
                    "acquisition": acquisition,
                    "dataset_name": dataset_name,
                    "iteration": i,
                    "n_top100": n_top100,
                    "n_cliff": n_cliff,
                    "n_both": n_both,
                    "n_cliff_orig": cliff_df["cliff_mol"].sum(),
                    "n_orig": len(cliff_df),
                    "n_cliff_top100_orig": cliff_df[cliff_df["top100"]]["cliff_mol"].sum()
                })
            collect.append(pd.DataFrame(results))

    collect = pd.concat(collect, ignore_index=True)
    collect.to_csv('cliff_analysis.csv', index=False)
else:
    collect = pd.read_csv('cliff_analysis.csv')


# Group by dataset_name, model, acquisition, loss, then take mean over runs (iteration)
plot_df = collect.groupby(["dataset_name", "model", "acquisition", "loss"]).agg({
    "n_top100": "mean",
    "n_cliff": "mean",
    "n_both": "mean",
    "n_cliff_top100_orig": "mean"
}).reset_index()

# Compute fractions
plot_df["frac_cliff_top100_orig"] = plot_df["n_cliff_top100_orig"] / 100
plot_df["frac_top100_found"] = plot_df["n_top100"] / 100
plot_df["frac_both"] = plot_df["n_both"] / 100

# Get unique values for model, acquisition, and loss
models = plot_df["model"].unique()
acquisitions = plot_df["acquisition"].unique()
losses = plot_df["loss"].unique()

# Create output directory for plots
output_dir = "cliff_analysis_plots"
os.makedirs(output_dir, exist_ok=True)

# Collect statistics for all model/acquisition/loss combinations
all_stats = []

for acquisition in acquisitions:
    for loss in losses:
        for model in models:
            # Filter the original collect DataFrame for this model/acquisition/loss
            sub_collect = collect[
                (collect["model"] == model) &
                (collect["acquisition"] == acquisition) &
                (collect["loss"] == loss)
            ]
            if sub_collect.empty:
                continue

            # Group by dataset_name and iteration to get per-run values
            grouped = sub_collect.groupby(["dataset_name", "iteration"]).agg({
                "n_top100": "mean",
                "n_both": "mean"
            }).reset_index()

            # Now, for each dataset_name, compute mean and 95% CI over iterations for the fractions
            def mean_ci(series):
                arr = np.array(series) / 100
                n = len(arr)
                mean = arr.mean()
                ci = 1.96 * arr.std() / np.sqrt(n)
                return mean, ci

            stats = grouped.groupby("dataset_name").agg(
                frac_top100_found_mean = ("n_top100", lambda x: mean_ci(x)[0]),
                frac_top100_found_ci = ("n_top100", lambda x: mean_ci(x)[1]),
                frac_both_mean = ("n_both", lambda x: mean_ci(x)[0]),
                frac_both_ci = ("n_both", lambda x: mean_ci(x)[1])
            ).reset_index()

            # Get frac_cliff_top100_orig for each dataset_name (from plot_df, which is already mean over runs)
            orig_stats = plot_df[
                (plot_df["model"] == model) &
                (plot_df["acquisition"] == acquisition) &
                (plot_df["loss"] == loss)
            ][["dataset_name", "frac_cliff_top100_orig"]].drop_duplicates()

            stats = stats.merge(orig_stats, on="dataset_name", how="left")

            # Save stats with model/acquisition/loss info for later aggregation
            stats["model"] = model
            stats["acquisition"] = acquisition
            stats["loss"] = loss
            all_stats.append(stats)

            labels = []
            bottoms = []
            bottoms_err = []
            tops = []
            remainders = []
            frac_top100_found_means = []
            frac_top100_found_cis = []

            stats = stats.sort_values('frac_cliff_top100_orig', ascending=False)

            for idx, row in stats.iterrows():
                labels.append(f"{row['dataset_name']}")
                frac_both = row["frac_both_mean"]
                frac_both_ci = row["frac_both_ci"]
                frac_top100_found = row["frac_top100_found_mean"]
                frac_top100_found_ci = row["frac_top100_found_ci"]
                frac_cliff_top100_orig = row["frac_cliff_top100_orig"]

                bottom = frac_both
                # Remove the "Found Top100" category from the stacked bar
                # Only show "Found Top100 Cliff" (bottom), "Top100 Cliff" (top), and remainder
                top = max(frac_cliff_top100_orig - frac_both, 0)
                remainder = max(1.0 - (bottom + top), 0)

                bottoms.append(bottom)
                bottoms_err.append(frac_both_ci)
                tops.append(top)
                remainders.append(remainder)
                frac_top100_found_means.append(frac_top100_found)
                frac_top100_found_cis.append(frac_top100_found_ci)

            ind = np.arange(len(labels))
            width = 0.5

            fig, ax = plt.subplots(figsize=(max(8, len(labels)*0.5), 8))

            # Plot the bars
            p1 = ax.bar(ind, bottoms, width, label="Cliff molecules found", color="#00496FFF", yerr=bottoms_err, capsize=4)
            p2 = ax.bar(ind, tops, width, bottom=bottoms, label="Cliff molecules", color="#0c6e85")
            p3 = ax.bar(ind, remainders, width, bottom=np.array(bottoms)+np.array(tops), color="#0F85A0FF")

            # Jitter should be small compared to bar width
            rng = np.random.default_rng(seed=42)
            jitter = rng.uniform(-0.15, 0.15, size=len(ind))  # width/2 = 0.35, so 0.15 is safe

            # Plot the fraction of found Top100 as a star marker (with error bar), with jitter
            ax.errorbar(
                ind + jitter,
                frac_top100_found_means,
                yerr=frac_top100_found_cis,
                fmt='o',
                color='#ccc640',
                markersize=6,
                elinewidth=1,
                capsize=3,
                label="Fraction of Top 100 found"
            )

            ax.set_ylabel("Fraction of top 100")
            # ax.set_title(f"MoleculeACE cliff molecules found by {model.upper()} {loss_map[loss]} with {acq_map[acquisition]}")
            ax.set_xticks(ind)
            ax.set_xticklabels(labels, rotation=70)
            ax.set_ylim(0, 1)
            ax.legend(loc="upper right")

            plt.tight_layout()
            # Save the plot as a PNG file
            fname = f"cliffbar_{model}_{acquisition}_{loss}.svg"
            fname = fname.replace("/", "-")  # sanitize filename if needed
            plt.savefig(os.path.join(output_dir, fname))
            plt.close(fig)



# After the loop, concatenate all stats and compute the average and standard error of (frac_both_mean / frac_cliff_top100_orig)
if all_stats:
    all_stats_df = pd.concat(all_stats, ignore_index=True)
    # Only keep rows where frac_cliff_top100_orig > 0 to avoid division by zero
    all_stats_df = all_stats_df[all_stats_df["frac_cliff_top100_orig"] > 0].copy()
    all_stats_df["frac_both_over_cliff"] = all_stats_df["frac_both_mean"] / all_stats_df["frac_cliff_top100_orig"]

    # Define the models and their display names
    violin_groups = [
        ("bnn", "ucb", "bnn"),
        ("gnn", "ucb", "gnn"),
        ("mlp", "greedy", "mlp"),
        ("gp", "ucb", "gp"),
    ]

    # For bnn, gnn, mlp: only "ranking" and "mse" losses; for gp: only "gp + mll"
    def get_group_label(row):
        for model, acq, disp in violin_groups:
            if row["model"] == model and row["acquisition"] == acq:
                return disp
        return None

    # Map loss to display label
    # loss_labels = {"ranking": "Ranking", "mse": "Regression", "gp + mll": "GP + MLL"}

    all_stats_df["group_label"] = all_stats_df.apply(get_group_label, axis=1)
    # all_stats_df["loss_label"] = all_stats_df["loss"].map(loss_labels)

    # Only keep rows that are in the selected groups
    plot_df = all_stats_df[all_stats_df["group_label"].notnull()].copy()

    # For bnn, gnn, mlp: only keep "ranking" and "mse"
    mask_non_gp = plot_df["group_label"].isin(["BNN + UCB", "GNN + UCB", "MLP + greedy"])
    plot_df = plot_df[~(mask_non_gp & (~plot_df["loss"].isin(["ranking", "mse"])))]

    # For gp: only keep "gp + mll"
    mask_gp = plot_df["group_label"] == "gp"
    # Where the model is "GP + UCB", set the loss to "gp + mll"
    plot_df.loc[mask_gp, "loss"] = "gp + mll"
    plot_df = plot_df[~(mask_gp & (plot_df["loss"] != "gp + mll"))]

    # Set up the order for plotting
    group_order = [g[2] for g in violin_groups]
    # For loss order: "Ranking", "Regression", "GP + MLL"
    loss_order = ["mse", "ranking", "gp + mll"]

    # Use seaborn's colorblind palette for the three loss types
    colorblind_palette = sns.color_palette("colorblind", n_colors=3)
    palette = dict(zip(loss_order, colorblind_palette))

    plt.figure()
    ax = sns.violinplot(
        data=plot_df,
        x="group_label",
        y="frac_both_over_cliff",
        hue="loss",
        order=group_order,
        hue_order=loss_order,
        palette=palette,
        cut=0,
        inner="box",
        linewidth=1.2,
        density_norm="width",
        width=0.7,
        split=False,
    )

    # Remove duplicate legend entries
    handles, labels_ = ax.get_legend_handles_labels()
    new_handles = []
    new_labels = []
    for h, l in zip(handles, labels_):
        if l not in new_labels:
            new_handles.append(h)
            new_labels.append(l)
    
    ax.legend(new_handles, new_labels, handlelength=0.5)

    ax.set_ylabel("Fraction of cliff molecules")
    ax.set_xlabel("")
    ax.set_ylim(0, 0.7)
    # ax.set_title("Fraction of cliff molecules found by model and loss")
    plt.xticks(rotation=0, ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cliff_fraction_violinplot.png"))
    plt.close()

import pdb; pdb.set_trace()
