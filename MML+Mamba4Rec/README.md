# MML+Mamba4Rec


## Specific steps
1. When the data_path and domains are changed, please modify the information of domains in `config_3.json`.
```
{
  "domains": [
    The names of all domains
  ],
  "data_path": The path of folder containing all datasets
}
```
2. If files do not exist in the folder (i.e.,`./Datasets/Amazon_platform/all`), please set "`--load_processed_data=false`" in `run.sh` for preparing the training, validation and testing datasets. 

   If processed datasets exist in the folder, please set "`--load_processed_data=true`" in `run.sh` for saving training time.
3. Conduct the grid search for some vital hyperparameters. This step can be skipped since the optimal hyperparamters have been saved in `run.sh`. 
```
bash search_parameters.sh
```
4. Run the MML+Mamba4Rec under the optimal hyperparamters.
```
bash run.sh 

```
