# tSZ_NG

This directory contains the core modules used in [Sabyr, Hill, and Haiman 2024]().

## Requirements

Modules can be used separately so have different requirements. In general, the following packages are needed:

[LensTools](https://github.com/asabyr/LensTools) (requires modified version),
[hmpdf](https://github.com/asabyr/hmpdf),
[numpy](https://numpy.org/),
[scipy](https://scipy.org/),
[pixell](https://pixell.readthedocs.io/en/latest/)

(optional, only needed for examples)
[matplotlib](https://matplotlib.org/),
[getdist](https://getdist.readthedocs.io/en/latest/)

(optional)
[NaMaster](https://namaster.readthedocs.io/en/latest/)

## Description

[code/forecast](code/forecast/): 
contains modules for computing forecasts. The main modules are ```fisher.py``` and ```constraints.py``` for fisher and suite-based forecasts, respectively. 
Some helper functions are in inference_funcs.py and fisher_funcs.py. 

[code/general](code/general/): 
contains various modules for analyzing maps and computing statistics. To compute various statistics across maps, you can use ```calc_stats.py``` with an ini file. 

```python calc_stats.py example.ini```

The script runs [code/general/analyze_maps.py](code/general/analyze_maps.py) according to the settings in [ini_files](ini_files/), which performs various map manipulations and computes/saves summary statistics. It relies on several modules: 

- split_maps.py -- cutting/trimming large maps into smaller ones, 
- map_utils.py -- for smoothing/filtering operations,
- noise.py -- making noise maps
- signal_noise.py -- for smoothing/filtering/generating noise maps, 
- namaster_flat.py -- computing power spectra using namaster.

Additional helper functions are in extra_funcs.py; read_stats.py is a module for processing a single summary statistic from a directory; rebin_peaks.py is for rebinning peak/minima counts.

[code/suite_code](code/suite_code/): 
- compute_avg_suite.py and compute_covariance_nsims.py -- for reading summary statistics and computing averages/covariances across the suite.
- read_combined_stats.py -- module for computing averages over individual directories.
- help_funcs.py -- helper functions for writing jobs/ini files

[code/convergence](code/convergence/): 
- stat_conv.py -- computing mean summary statistics as a function of the number of simulations
- cov_conv.py and compute_cov_nsims.py -- compute convergence of covariance elements and compute covariance for some number of sims
- check_maps_dir.py -- find any maps with trails, if the cluster run was not stable
- nsims_convergence_funcs.py -- functions for testing convergence of constraints wrt to mean stat/covariance

[code/examples](code/examples/):
This folder includes some examples for running parts of the code + post-processing results. 

- fisher_masses.py -- example to compute Fisher constraints + convergence checks
- convergence_stats.py -- compute tests for mean/covariance convergence
- MF_tests.py -- script to write job/ini files to compute Minkowski functionals for different set-ups
- moments_cosmo.py -- compute stat convergence wrt to number of simulations
- example_constraints.ipynb -- example to compute Fisher constraints + from the suite

Some example job files are also included. 

## Notes
Most modules are general and can be used for forecasts/computing summary statistics directly on a given simulation suite and/or with easy modifications to the code. If you use this code, please cite the paper.

## Aknowledgements 

Some parts of the code are adapted from LensTools. Thank you to Leander Thiele for the help with running hmpdf. 
