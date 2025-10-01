# MML

The source code for our Paper [**"Multi-Domain Sequential Recommendation via Multi-sequence  and Multi-task Learning"**].


## Environment

Our code is based on the following packages:
- GPU: Tesla V100-PCIe
- Requirments： 
   - Python 3.9.12
   - PyTorch 2.2.1
   - tqdm 4.65.0
   - pandas 1.4.2
   - numpy 1.19.2
   - mamba-ssm 1.2.0.post1
   - causal-conv1d 1.2.0.post2


## Usage

1. Download the datasets and put the files in `Datasets/`.

2. Run the data preprocessing scripts to generate the data. 
``` 
cd Datasets
python process.py 
python process_dataset_later.py
```
More details on data processing can be found in `Datasets/README.md`.

3. To run the program, try the script given in 'run.sh' in the corresponding folder.
``` 
cd MML+GRU4Rec
bash run.sh 
cd MML+SASRec4Rec
bash run.sh 
cd MML+Mamba4Rec
bash run.sh 
```

## Cite

If you find this repo useful, please cite
```
@article{MML,
	title={Multi-Domain Sequential Recommendation via Multi-sequence and Multi-task Learning},
	author={Liwei Pan and Weike Pan and Zhong Ming},
	journal={Information Processing \& Management},
	publisher={Elsevier}
}
```
