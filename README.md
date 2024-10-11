# Rank-based Bayesian optimization (RBO) for molecular datasets
Using ranking models in the context of Bayesian optimization of materials and chemical discovery.

## Creating environment
The code is implemented in Python, and utilizes a conda environment. 
```bash
conda create --name rbo --file environment.yml
```

Alternatively, the environment can be manually created:
```bash
# create and activate
conda create -n rbo python=3.8 -y
conda activate rbo

# besure to install pytorch with correct cuda version before pytorch-scatter
conda install -c pytorch -c nvidia pytorch-cuda=12.1 -y
conda install -c pyg pyg=*=*cu* -y
conda install -c gpytorch gpytorch -y

# install conda packages first
conda install -c conda-forge scikit-learn pandas seaborn matplotlib scipy numpy -y
conda install -c conda-forge rdkit=2024.03.1 -y

# install pip packages next
pip install rogi
pip install PyTDC
pip install gauche bayesian-torch --no-deps     # suppress dependencies to avoid 
                                                # affecting conda installs of pytorch

# deactivate
conda deactivate 
```

## Running the code
Scripts for running the experiments are found in `scripts` folder. The reproduce the results in the manuscript, run:
```bash
python main.py \
    --dataset_name=<dataset> \
    --rank \                             # toggle the usage of ranking model
    --maximize \                         # toggle maximization of target value
    --use_gpu \                          # toggle the usage of GPU, will use CPU if none found
    --scale \                            # toggle the robust scaling of targets
    --model_type=<model> \
    --acq_func=<acquisition function> \
    --num_init=10 \
    --num_epochs=100 \
    --batch_size=5 \
    --budget=100 \
    --num_runs=20
```
More options are available in the `main.py` script. These are the values used in the experiments presented in manuscript.

## Analysis of results
Scripts used to analyze the results and generate the plots are also presented in the `scripts` folder. 

The file `scripts/rogi_analysis.py` assumes that all results have been calculated for the ZINC datasets (all 12 dataset, 4 models, 3 acquisition functions).

Similarily, `scripts/chembl_analysis.py` assumes that either all the Ki datasets or all the EC50 datasets in MoleculeACE have already been tested with RBO (4 models, and 3 acquisition functions).

All statistics will be printed in and saved in latex files, and the plots will be generated as png in the working directory.


