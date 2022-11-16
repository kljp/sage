# SAGE: toward on-the-fly gradient compression ratio scaling

## Description
> ***SAGE*** is abbreviation of 'Sparsity-Adjustable Gradient Exchanger'. SAGE supports dynamic scaling of gradient compression ratio on-the-fly. Benchmarks include: 1) image classification using CNNs, 2) language modelling using LSTMs `cnn-lstm`, and 3) recommendation using NCF `ncf`.

## How to setup

To install the necessary dependencies, create Conda environment using `environment.yml` by running the following commands. ***Note***: check compatibility of each package version such as `cudatoolkit` and `cudnn` with your device, e.g., NVIDIA Tesla V100 GPU is compatible.

```bash
$ conda env create --file environment.yml
$ conda activate sage
$ python -m spacy download en
$ conda deactivate sage
```

## How to start

The scripts to run code are written for SLURM workload manager, and setup of **multi-node (each with 1 GPU)**. In `run.sh`, you can specify *model*, *dataset*, ***reducer***, and *world_size*. ***Note***: if you want to use multiple GPUs for each node, use **mpirun** to execute `run.py` in `run.sh`; but, you should modify source code to give correct rank to each prcocess.
- If you use **SLURM**, use `pararun` and modify it for your configuration. The script `pararun` executes `run.sh` in parallel. The script `run.sh` includes setup for distributed training.
- If you do not use SLURM, you do not need to use `pararun`. Instead, run `run.sh` on your nodes, then rendezvous of pytorch allows processes are connected. For this distributed training, you should modify three lines of code: specify ***RANK***, ***hostip***, ***port*** to use in `run.sh`.

- ### CNN and LSTM benchmarks

 #### - Run training script with dataset download

 - If you use **SLURM**, use following command.
```bash
$ sbatch pararun
```
 - If you do not use SLURM, use following command on each node.
```bash
$ hostip=<ip> port=<port> rank=<rank> ./run.sh
```

- ### Neural Collaborative Filtering (NCF) benchmarks

 #### 1. Prepare dataset

 - To download dataset, use following command.
```bash
$ ./prepare_dataset.sh
```

 #### 2. Run training script

 - If you use **SLURM**, use following command.
```bash
$ sbatch pararun
```
 - If you do not use SLURM, use following command on each node.
```bash
$ hostip=<ip> port=<port> rank=<rank> ./run.sh
```

## Acknowledgements

Most of code except [SAGE](https://github.com/kljp/sage) implementation is provided by previous works. If you use the code, please cite the following papers also.

**PowerSGD** \[[Paper](https://arxiv.org/abs/1905.13727)\] \[[Code](https://github.com/epfml/powersgd)\] (`cnn-lstm`)

    @inproceedings{vkj2019powerSGD,
      author = {Vogels, Thijs and Karimireddy, Sai Praneeth and Jaggi, Martin},
      title = "{{PowerSGD}: Practical Low-Rank Gradient Compression for Distributed Optimization}",
      booktitle = {NeurIPS 2019 - Advances in Neural Information Processing Systems},
      year = 2019,
      url = {https://arxiv.org/abs/1905.13727}
    }
**Rethinking-sparsification** \[[Paper](https://arxiv.org/abs/2108.00951)\] \[[Code](https://github.com/sands-lab/rethinking-sparsification)\] (`cnn-lstm` and `ncf`)

    @inproceedings{sda+2021rethinking-sparsification,
      author = {Sahu, Atal Narayan and Dutta, Aritra and Abdelmoniem, Ahmed M. and Banerjee, Trambak and Canini, Marco and Kalnis, Panos},
      title = "{Rethinking gradient sparsification as total error minimization}",
      booktitle = {NeurIPS 2021 - Advances in Neural Information Processing Systems},
      year = 2021,
      url = {https://arxiv.org/abs/2108.00951}
    }

## Publication

No publication yet.

## Contact

If you have any questions about this project, contact me by one of the followings:
- slashxp@naver.com
- kljp@ajou.ac.kr
