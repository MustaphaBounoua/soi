[![arXiv](https://img.shields.io/badge/arXiv-2402.05667-b31b1b.svg)](https://arxiv.org/pdf/2402.05667)

[![Venue](https://img.shields.io/badge/venue-ICML_2024-darkblue)](https://icml.cc/virtual/2024/oral/35535)
[![Venue](https://img.shields.io/badge/Oral-presentation-darkred)](https://icml.cc/virtual/2024/oral/35535)



# S*\Omega*I: Score-based O-INFORMATION Estimation

This repository contains the implementation for the paper [Score-based O-INFORMATION Estimation](https://arxiv.org/pdf/2402.05667) presented at ICML 2024.

## Table of Contents

- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contact](#contact)

## Description
The repository provides the source code and datasets used in the research paper "S*\Omega*I : Score-based O-INFORMATION Estimation". This is a tool to analyse multivariate information. S*\Omega*I is a new method to estimate O-information using score functions to describe high interdependencies in complex systems. We show its effectiveness on synthetic data and a real neural application.

### Installation
```bash
pip install -r requirements.txt
```

## Usage


### Demo

#### Synthetic benchmark

The files in '\demo\Quickstart' presents a straightforward quickstart to use *S\Omega*I. 
First, by default config can be loaded :
```
    args=get_config().parse_args([])
```

A synthetic benchmark can be created : 
```
args.dim = 1
args.rho = 0.5
my_settings = [{"rho":0.6,"type":"red","nb":3},{"rho":0.4,"type":"syn","nb":3} ]
task = get_task(args,my_settings)
```
Groud truth information measures can be obtaining:

```
task.get_summary()
```

#### Runing S*\Omega*I

- First, obtain the data loaders :
```
train_l, test_l  = get_dataloader(task,args)
```
- Instantiante SOI object

```
soi = SOI(args, nb_var = task.nb_var, test_loader=test_l, gt = task.get_summary())
```
Fit the model
```
soi.fit(train_l, test_l)
```

Compute O_information using the test_loader

```
soi.compute_o_inf_batch(test_l)
```



### Running the Code

To run  a particular experiment 


To run the experiments and reproduce the results from the paper, the shell scripts containing all the configuration are in '\jobs':



### Configuration
The configurations for different experiments are stored in the `experiments/config.py` file. 

### Datasets

Datasets used in the experiments are stored in the `data` directory. Ensure that the necessary data files are available before running the experiments.

## Project Structure
```
MLD/
├── data/                  # Directory containing VBN datasets after runing src/vbn/downoalad.py
├── experiments/           
│   ├── run_soi.py         # run SOI experiment on synthetic benchmark
│   ├── run_soi_grad.py    # run SOI experiment on synthetic benchmark and estimate gradient of O-information
│   ├── run_soi_vbn.py     # run SOI experiment on the VBN dataset
│   └── run_baseline.py    # run baselines on synthetic benchmark
├── src/                   # Source code for the project
│   ├── models/            # Denoising score models : MLP with skip connection and a transformer implementation
│   ├── benchmark/         # Synthetic benchmark
│   └── libs               # SOI models
│       ├──soi.py          # The main SOI model class
│       ├──soi_grad.py     # A subclass of the soi model that takes into account gradint of o-information
│       ├──SDE.py          # The noising process which permits the learn the score functions
│       ├──inf_measures.py # The set of functions to compute all the information measures: O-information, S-information, TC, DTC .. etc
│       ├──importance.py   # Required function to implement importance sampling scheme.
│       └──util.py         # General utility functions
├── requirements.txt       # List of dependencies
└── README.md              # This README file
```

## Contact
For any questions or inquiries, please contact Mustapha Bounoua at mustaphabounoua96@gmail.com.

## Cite our paper