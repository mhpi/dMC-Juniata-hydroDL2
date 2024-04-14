<img alt="image" src="https://github.com/mhpi/dMC-Juniata-hydroDL2/assets/16233925/c93f352c-648d-40cb-bee7-2ee8916d4e89">


# dMC-Juniata-hydroDL2

[![DOI](https://zenodo.org/badge/719824272.svg)](https://zenodo.org/doi/10.5281/zenodo.10183448)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)
![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)
![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)

This repo contains a released version of the differentiable Muskingum-Cunge Method from https://doi.org/10.1029/2023WR035337

## Installation/Getting Started

1. Create your environment

Use Conda to create an environment

```shell
conda env create -f environment.yml
```

Or use a combination of conda + pip

```shell
conda create -n dMC-Juniata-hydroDL2
conda activate dMC-Juniata-hydroDL2
pip install -r requirements.txt
```

2. Download the River graph data from our Zenodo link https://zenodo.org/records/10183429 for the Susquehanna River Basin

3. Run an experiment. Your experiments are controlled by config files within the `dMC/conf/configs` dir.

To change the config file, go to `dMC/conf/global_settings.yaml` and make sure to change the experiment `name` as desired

## How to use this package:
dMC Routing is composed of several class objects:
- `DataLoader`
  - Contains all data information, and is an iterable to assist with training and validation.
  - The dataloader provided is for data in the Susquehanna River Basin
- `Model`
  - The differentiable routing model
- `Experiment`
  - The experiment you are going to run. 
  - Your experiment is your use case for this class. Say you want to train an MLP, there is an experiment for that. Or you want to generate 
synthetic data... there is an experiment for that.

This code is set up so your experiment file is similar to a script, but all of the function imports and class creations are done behind the scenes
providing a cleaner, abstract, interface.

When running the code from the cmd line: `python -m dMC`, these classes are instantiated from the `factory.py` file
and are run in your experiment. 

Inside of every config file there is a `Service_Locator`
```yaml
# service_locator -----------------------------------------------------------------------
service_locator:
  experiment: generate_synthetic.GenerateSynthetic
  data: nhd_srb.NHDSRB
  observations: usgs.USGS
  physics: explicit_mc.ExplicitMC
  neural_network: single_parameters.SingleParameters
```
This config entry will point to the `file_name.class_name` imported behind the scenes. 

## Experiments

To run an experiment from the command line: You need to set up the `dMC/conf/global_settings.yaml` file. 
This file includes the following default information:
```yaml
cwd: /path/to/your/codefolder/dMC-Juniata-hydroDL2
data_dir: /path/to/your/data/dx-2000dis1_non_merge
name: config file name
device: cpu
```
This information will be global to all experiments, and is set outside of the individual config files
- `cwd`
  - The current working directory where you cloned this repo. For example, my `cwd` is `/home/tbindas/dMC-Juniata-hydroDL2`
- `data_dir`
  - The directory that you downloaded the Zenodo Data to, or where your graph data lives. Mine is `/data/dx-2000dis1_non_merge`
- `name`
  - The name of your experiment run. I always name these after the experiment I'm running
- `device`
  - Currently only CPU is supported

On top of the `global_settings.yaml` file is:
```yaml
defaults:
  - config: 03_train_usgs_period_1a
```
This is where you specify the experiment config that you would like to run. See below for an organization of all experiment files:
### 01: Single Parameter Experiments
To run these, you should use the following configs:
- `01_generate_single_synth_parameter_data.yaml`
- `01_train_against_single_synthetic.yaml`

### 02: Synthetic Parameter Distribution Recovery

There are many synthetic parameter experiments. Run the following configs to recreate them

#### Synthetic Constants
- `02_generate_mlp_param_list.yaml`
- `02_train_mlp_param_list.yaml`

#### Synthetic Power Law A
- `02_generate_mlp_power_a.yaml`
- `02_train_mlp_power_a.yaml`

#### Synthetic Power Law B
- `02_train_mlp_power_b.yaml`
- `02_generate_mlp_power_b.yaml`

### 03: Train against USGS data:
You can run the following cfgs to train models against USGS data
- `03_train_usgs_period_1a.yaml`
- `03_train_usgs_period_1b.yaml`
- `03_train_usgs_period_2a.yaml`
- `03_train_usgs_period_2b.yaml`
- `03_train_usgs_period_3a.yaml`
- `03_train_usgs_period_3b.yaml`
- `03_train_usgs_period_4a.yaml`
- `03_train_usgs_period_4b.yaml`

### Running experiments from a Juypter Notebook 
- See the `notebooks/` dir for a detailed example of how to use this repo in a Notebook setting with the provided configurations!

## Outputs:
Since we use Hydra, our output logs, config file, and saved data will be in the `dMC/outputs/` dir. The outputs are sorted by
the date the job was run (YYYY-mm-dd), and then the time the job was run (hh-mm-ss).

# Citation:
```bibtex
@article{https://doi.org/10.1029/2023WR035337,
author = {Bindas, Tadd and Tsai, Wen-Ping and Liu, Jiangtao and Rahmani, Farshid and Feng, Dapeng and Bian, Yuchen and Lawson, Kathryn and Shen, Chaopeng},
title = {Improving River Routing Using a Differentiable Muskingum-Cunge Model and Physics-Informed Machine Learning},
journal = {Water Resources Research},
volume = {60},
number = {1},
pages = {e2023WR035337},
keywords = {flood, routing, deep learning, physics-informed machine learning, Manning's roughness},
doi = {https://doi.org/10.1029/2023WR035337},
url = {https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2023WR035337},
eprint = {https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2023WR035337},
note = {e2023WR035337 2023WR035337},
abstract = {Abstract Recently, rainfall-runoff simulations in small headwater basins have been improved by methodological advances such as deep neural networks (NNs) and hybrid physics-NN models—particularly, a genre called differentiable modeling that intermingles NNs with physics to learn relationships between variables. However, hydrologic routing simulations, necessary for simulating floods in stem rivers downstream of large heterogeneous basins, had not yet benefited from these advances and it was unclear if the routing process could be improved via coupled NNs. We present a novel differentiable routing method (δMC-Juniata-hydroDL2) that mimics the classical Muskingum-Cunge routing model over a river network but embeds an NN to infer parameterizations for Manning's roughness (n) and channel geometries from raw reach-scale attributes like catchment areas and sinuosity. The NN was trained solely on downstream hydrographs. Synthetic experiments show that while the channel geometry parameter was unidentifiable, n can be identified with moderate precision. With real-world data, the trained differentiable routing model produced more accurate long-term routing results for both the training gage and untrained inner gages for larger subbasins (>2,000 km2) than either a machine learning model assuming homogeneity, or simply using the sum of runoff from subbasins. The n parameterization trained on short periods gave high performance in other periods, despite significant errors in runoff inputs. The learned n pattern was consistent with literature expectations, demonstrating the framework's potential for knowledge discovery, but the absolute values can vary depending on training periods. The trained n parameterization can be coupled with traditional models to improve national-scale hydrologic flood simulations.},
year = {2024}
}
```




