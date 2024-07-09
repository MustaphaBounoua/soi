[![arXiv](https://img.shields.io/badge/arXiv-2402.05667-b31b1b.svg)](https://arxiv.org/pdf/2402.05667)
[![Venue](https://img.shields.io/badge/venue-ICML_2024-darkblue)](https://icml.cc/virtual/2024/oral/35535)
[![Venue](https://img.shields.io/badge/Oral-presentation-yellow)](https://icml.cc/virtual/2024/oral/35535)



# S*\Omega*I: Score-based O-INFORMATION Estimation

This repository contains the implementation for the paper [Score-based O-INFORMATION Estimation](https://arxiv.org/pdf/2402.05667) presented at ICML 2024.

## Description
S*\Omega*I is a new method to estimate O-information using score functions to describe high interdependencies in complex systems. We show its effectiveness on synthetic data and a real neural application.

### Installation
```bash
pip install -r requirements.txt
```

## Usage


### Demo

#### Synthetic benchmark

The files in '\demo\Quickstart' presents a straightforward quickstart to use *S\Omega*I. 
First, by default config can be loaded :
```python
    args=get_config().parse_args([])
```

A synthetic benchmark can be created : 
```python
args.dim = 1
args.rho = 0.5
my_settings = [{"rho":0.6,"type":"red","nb":3},{"rho":0.4,"type":"syn","nb":3} ]
task = get_task(args,my_settings)
```
Groud truth information measures can be obtaining:

```python
task.get_summary()
```

#### Runing S*\Omega*I

- First, obtain the data loaders :
```python
train_l, test_l  = get_dataloader(task,args)
```
- Instantiante SOI object

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



### Running the Code

To run  a particular experiment 


To run the experiments and reproduce the results from the paper, the shell scripts containing all the configuration are in '\jobs':



### Configuration
The configurations for different experiments are stored in the `experiments/config.py` file. 

### Datasets

Datasets used in the experiments are stored in the `data` directory. Ensure that the necessary data files are available before running the experiments.

## Project Structure
```bash
MLD/
├── data/                  # VBN datasets after running src/vbn/downoalad.py
├── experiments/           
│   ├── run_soi.py         # Run SOI experiment on synthetic benchmark
│   ├── run_soi_grad.py    # Run SOI experiment on synthetic benchmark and estimate gradient of O-information
│   ├── run_soi_vbn.py     # Run SOI experiment on the VBN dataset
│   └── run_baseline.py    # Run baselines on synthetic benchmark
├── src/                   # Source code for the project
│   ├── models/            # Denoising score models: MLP with skip connection and a transformer implementation
│   ├── benchmark/         # Synthetic benchmark
│   └── libs               # SOI models
│       ├── soi.py         # Main SOI model class
│       ├── soi_grad.py    # Subclass of the SOI model that considers gradient of O-information
│       ├── SDE.py         # Noising process for learning score functions
│       ├── inf_measures.py # Functions to compute information measures: O-information, S-information, TC, DTC, etc.
│       ├── importance.py   # Function for implementing importance sampling scheme
│       └── util.py         # General utility functions
├── jobs                   # Useful scripts to reproduce paper results
├── demo                   # Jupyter notebook demonstrating how to use SOI 
├── requirements.txt       # List of dependencies
└── README.md              # This README file
```

## Contact
For any questions or inquiries, please contact Mustapha Bounoua at mustaphabounoua96@gmail.com.

## Cite our paper
```bibtex
@article{bounoua2024score,
  title={Score-based O-INFORMATION Estimation},
  author={Bounoua, Mustapha},
  journal={arXiv preprint arXiv:2402.05667},
  year={2024}
}
```
