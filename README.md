# LESS DPO data selection

## Install Requirements
**Environmnet**: To get started with this repository, you may need to install a conda environment based on our environment.yml file.
```
conda env create -f environment.yml

pip3 install fast-jl==0.1.3
```


## Run the less/less_dpo data selection pipline
For run the less and less_dpo data selection, you can directly run run_less_and_less_dpo.sh file as following:

```bash
cd ./run

data_seed=42  # Experiment seed
percentage=0.05   # Data percentage to select
model_load_path=model_path    # Your base LLM architechture here
devices="0,1,2,3,4,5,6,7"  # Cuda devices available to use
max_collect_samples=500  # The number of training data you want to test the code, after everything works, you can set it to None to run on all training data
projection_dims=8192  # The projection dimension

./run_less_and_less_dpo.sh "$data_seed" "$percentage" "$model_load_path" "$devices" "$max_collect_samples" "$projection_dims"
```
