# Fourier Value Functions

## Overview
Fourier Value Functions (FVF) are a framework for leverging Fourier-based techniques for value function approximation for use in policy learning. More preciesly, FVFs are SO(2)/SO(3)-equivariant models designed to
map 2D/3D environmental geometries to continuous state-action values. FVFs operates entirely in Fourier space, encoding geometric structure into latent Fourier features using equivariant neural networks and then 
outputting the Fourier coefficients of the output signal. Combining these coefficients with harmonic basis functions enables the simultaneous prediction of all values of the continuous action space at any resolution.


### PushT
<p align="center">
  <img src="FVF_PushT.gif"/>
</p>

## Install
1. Install (EquiHarmoncy)[https://github.com/ColinKohler/EquiHarmony]

2. Clone this repository
```
git clone git@github.com:ColinKohler/fourier_value_functions.git
```
3. Install dependencies from Pipfile
```
pip install -r requirments.txt
```

## Running
```
python scripts/train.py --config-name=train_polar_fvf_lowdim_workflow.yaml
```

Additional config files can be found in `fourier_value_functions/config/`.

## Cite
