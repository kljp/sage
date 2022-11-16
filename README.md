# SAGE: toward on-the-fly gradient compression ratio scaling

## Description
> ***SAGE*** is abbreviation of 'Sparsity-Adjustable Gradient Exchanger'. SAGE supports dynamic scaling of gradient compression ratio on-the-fly. Benchmarks include: 1) Image Classification using CNNs, 2) Language Modelling using LSTMs `cnn-lstm`, and 3) Recommendation using NCF `ncf`.

## How to setup

To install the necessary dependencies, create Conda environment using `environment.yml` by running the following commands (note: check compatibility of each package version such as `cudatoolkit` and `cudnn` with your device, e.g., NVIDIA Tesla V100 GPU is compatible).

```bash
$ conda env create --file environment.yml
$ conda activate sage
$ python -m spacy download en
$ conda deactivate sage
```

## How to start
TBD
### CNN and LSTM benchmarks

TBD

### Neural Collaborative Filtering (NCF) benchmarks
#### 1. Prepare dataset
TBD
#### 2. Run training process
TBD

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
