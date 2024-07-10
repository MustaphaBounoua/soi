[![arXiv](https://img.shields.io/badge/arXiv-2402.05667-b31b1b.svg)](https://arxiv.org/pdf/2402.05667)
[![Venue](https://img.shields.io/badge/venue-ICML_2024-darkblue)](https://icml.cc/virtual/2024/oral/35535)
[![Venue](https://img.shields.io/badge/Oral-presentation-darkred)](https://icml.cc/virtual/2024/oral/35535)



# SΩI: Score-based O-INFORMATION Estimation

This repository contains the implementation for the paper [SΩI : Score-based O-INFORMATION Estimation](https://arxiv.org/pdf/2402.05667) presented at ICML 2024.


## Description
The analysis of scientific data and complex multivariate systems requires information quantities that capture relationships among multiple random variables. Recently, new information-theoretic measures have been developed to overcome the shortcomings of classical ones, such as mutual information, that are restricted to considering pairwise interactions. Among them, the concept of information synergy and redundancy is crucial for understanding the high-order dependencies between variables. One of the most prominent and versatile measures based on this concept is O-information, which provides a clear and scalable way to quantify the synergy-redundancy balance in multivariate systems. However, its practical application is limited to simplified cases. In this work, we introduce SΩI, which allows to compute O-information without restrictive assumptions about the system while leveraging a unique model. Our experiments validate our approach on synthetic data, and demonstrate the effectiveness of SΩI in the context of a real-world use case.

## Installation
```bash
pip install -r requirements.txt
```

## Usage


### Demo

Checkout  `Quickstart.ipynb` for a quickstart on how to use SΩI.



First, default config can be loaded :
```python
args=get_default_config()
```

A synthetic benchmark can be created : 
```python
my_settings = [{"rho":0.6,"type":"red","nb":3},{"rho":0.6,"type":"Syn","nb":3} ]
args.dim = 1
task = get_task(args,my_settings)

```
Groud truth information measures can be obtained:

```python 
task.get_summary()
```

#### Runing SΩI

Build the dataloaders 

```python
train_l, test_l  = get_dataloader(task,args)
```

Instantiante SOI object

```python
soi = SOI(args, nb_var = task.nb_var, test_loader=test_l, gt = task.get_summary())
```
Fit the model

```python
soi.fit(train_l, test_l)
```

Compute O_information using the test_loader

```python
soi.compute_o_inf_batch(test_l)
```



### Running experiments

Running a particular experiment can be done using the scripts in `experiments\`. The experiments configurations are described in `experiments\config.py`

Example:
```bash
python -m experiments.run_soi --arch "tx" --bechmark "red" --dim 1 --setting "0" --rho 0.4 --bs 256 --lr 0.001 --max_epochs 200 --use_ema --importance_sampling  --weight_s_functions
```

To run the experiments and reproduce the results from the paper, the shell scripts are provided in `jobs\`.

### Dataset

The synthetic benchmark are generated and not stored. The VBN dataset used in the neural activity experiments are stored in the `data/` directory.  The preprocessed VBN dataset can be generated by running `src/vbn/download`. Please note that this script will require to download the whole VBN dataset (3TB of storage).

## Project Structure
```
MLD/
├── experiments/           
│   ├── run_soi.py         # run SOI experiment on synthetic benchmark
│   ├── run_soi_grad.py    # run SOI experiment on synthetic benchmark and estimate gradient of O-information
│   ├── run_soi_vbn.py     # run SOI experiment on the VBN dataset
│   └── run_baseline.py    # run baselines on synthetic benchmark
├── src/                   # Source code for the project
│   ├── models/            # Denoising score models : MLP with skip connection and a transformer implementation
│   ├── benchmark/         # Synthetic benchmark
│   ├── baseline           # baseline constructed using MI neural estimators
│   ├── vis                # visualization notebook
│   └── libs               # SOI models
│       ├──soi.py          # The main SOI model class
│       ├──soi_grad.py     # A subclass of the soi model that takes into account gradint of o-information
│       ├──SDE.py          # The noising process which permits the learn the score functions
│       ├──inf_measures.py # The set of functions to compute all the information measures: O-information, S-information, TC, DTC .. etc
│       ├──importance.py   # Required function to implement importance sampling scheme.
│       └──util.py         # General utility functions
├── jobs                   # bash script to reproduce the paper results
├── quickstart.ipynb       # a jupyter notebook which explain how to use soi                           
├── requirements.txt       # List of dependencies
└── README.md              # This README file
```



## Cite our paper

```bibtex
@inproceedings{
bounoua2024somegai,
title={S\${\textbackslash}Omega\$I: Score-based O-{INFORMATION} Estimation},
author={Mustapha BOUNOUA and Giulio Franzese and Pietro Michiardi},
booktitle={Forty-first International Conference on Machine Learning},
year={2024},
url={https://openreview.net/forum?id=LuhWZ2oJ5L}
}
```