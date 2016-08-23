# Matching tutor to student: rules and mechanisms for efficient two-stage learning in neural circuits
Authors: Tiberiu Tesileanu, Bence Olveczky, Vijay Balasubramanian

This repository contains the code and data for our paper on efficient two-stage learning in songbirds (and beyond).

Most of the detailed simulation code is contained in the files `simulation.py` and `basic_defs.py`. There are tests checking that the code works properly, contained in `tests.py`. It's a good idea to run this script first after downloading the code to make sure that everything is set up correctly.

The code uses Python 2.7, and it requires a relatively recent installation of `iPython` (including the `jupyter` notebook), `matplotlib`, `numpy`, `scipy`, and `seaborn`. The optimization code uses CMA-ES optimization routines that can be downloaded from https://www.lri.fr/~hansen/cmaes_inmatlab.html.

The code responsible for generating the results and making the plots from the paper is contained in iPython notebooks `rate_based_simulations.ipynb` for the rate-based model and `spiking_simulations.ipynb` for the spiking model. The spiking model makes use of the parameters obtained from the optimization procedure described in the Methods section of our paper; these parameters are available in `default_params.pkl`. The code makes use of `helpers.py`, which contains various functions that are useful for visualizing the results of the simulations.

To perform the parameter optimization, use the `experiment_matcher.ipynb` notebook. The data used for the optimization is contained in the `data` folder -- we thank Timothy Otchy for parts of this data. Note that due to the stochastic nature of the learning simulations and of the CMA-ES optimization algorithm, the result will change every time this code is run. This means that you will not get exactly the same parameters as in `default_params.pkl` upon running this code.

`plasticity_plot.ipynb` is a short notebook that was used to make the plot of the plasticity curve for our rule when `alpha = beta = 1`.

Results and figures are saved in the `save` and `figs` folders, respectively, by the iPython notebooks. `run_once.py`, `run_reinf_rate_optim.py`, `run_reinf_tscale.py`, and `run_tscale_batch.py` are scripts that can be used to generate the job scripts necessary to run the time-consuming spiking simulations on a cluster. These assume a system based on `qsub` and `bash`. `summarize.py` can be used to 'summarize' the results from multiple batch runs by keeping only information about the learning curve and deleting the (very space-consuming) information about intermediate states of the learning process.

## Contact
If you have any issues or questions regarding the code, please use the issue tracker, or write to Tiberiu Tesileanu at ttesileanu@gmail.com.
