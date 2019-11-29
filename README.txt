This project contains files with functions written in python related to directional predictions, simulation of synthetic samples and directional analyses based on the simplified zonal giant Gaussian process models presented by Brandt et al. (in prep.).

The synthetic simulations part needs the previous installation of the PMAGPY package.

The files that contains the main functions are:

1) PSV_dir_predictions.py:
	Contains theoretical determinations for variances and covariances for a given zonal GGP model, the pdf map of the GGP model s(u) in equal-area coordinates, mean values, standard deviations. 
	A jupyter notebook "Directional_PSV_analises_functionsfromBrandtetal2019.ipynb" was created to help the user know more about these functions.

2) Synthetic_directions_GGP.py
	Contains functions for simulating synthetic directions and calculating the PSV directional measurements.
	This file needs the previous instalation of PMAGPY package (https://github.com/PmagPy/PmagPy). 	
	A jupyter notebook "Simulation of synthetic directions from zonal GGP models and experimental results.ipynb" was created to help the user know more about these functions.

