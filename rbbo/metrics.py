import pandas as pd
from sklearn.metrics import auc 

from .utils import min_max_scale, remove_outliers

def frac_top_n(
    df: pd.DataFrame, 
    bo_output: pd.DataFrame, 
    n: int, 
    goal: str = 'maximize'
):
    if goal == "maximize":
        df = df.nlargest(n, "target", keep="first")
    elif goal == "minimize":
        df = df.nsmallest(n, "target", keep="first")

    count = 0
    fracs = []
    for index, row in bo_output.iterrows():
        if row["smiles"] in df["smiles"].tolist():
            count += 1
        frac = count / float(n)
        fracs.append(frac)
    bo_output["frac_top_n"] = fracs

    return bo_output


def top_one(
    bo_output: pd.DataFrame, 
    goal: str = 'maximize'
):
    targets = bo_output["target"]
    if goal == 'maximize':
        bo_output["top_one"] = targets.cummax()
    elif goal == 'minimize':
        bo_output["top_one"] = targets.cummin()
    return bo_output

def frac_top_n_percent(
        df: pd.DataFrame, 
        bo_output: pd.DataFrame, 
        n: int, 
        goal: str = 'maximize'
):

    if goal == 'maximize':
        q = 1 - (n/100)
        quantile = df['target'].quantile(q)
        df = df[df["target"] > quantile]
    elif goal == 'minimize':
        q = n/100
        quantile = df['target'].quantile(q)
        df = df[df["target"] < quantile]

    count = 0
    fracs = []
    length = len(df)
    for index, row in bo_output.iterrows():
        if row["smiles"] in df["smiles"].tolist():
            count += 1
        frac = count / float(length)
        fracs.append(frac)
    bo_output["frac_top_n_percent"] = fracs

    return bo_output

def auc_metric(
        df: pd.DataFrame, 
        bo_output: pd.DataFrame, 
        metric: str = "top_one",
        goal: str = "maximize",
        outliers: float = None
):  
    # remove outliers if necessary
    if outliers is not None:
        df = remove_outliers(df, goal, num_sigma=outliers)

    # best and worst inside dataset
    if goal == 'minimize':
        best_val = df['target'].min()
        worst_val = df['target'].max()
    else:
        best_val = df['target'].max()
        worst_val = df['target'].min()

    # we will need to remove values that returned outside the best and worst
    if outliers:
        if goal == 'minimize':
            bo_output.loc[bo_output[metric] > worst_val, metric] = worst_val
        elif goal == 'maximize':
            bo_output.loc[bo_output[metric] < worst_val, metric] = worst_val


    x = min_max_scale(bo_output['evaluation'])   # scale it to [0,1] for evaluations
    y = bo_output[metric]
    # y = min_max_scale(y, worst_val, best_val)

    custom_auc = auc(x, y)

    # if goal == 'minimize':
    #     custom_auc = 1.0 - custom_auc

    return custom_auc