## Dynamic Functional Connectivity Analysis with Sliding-Window Correlations, Dimensionality Reduction and k-means Clustering

This code runs sliding-window correlations (SWC) on preprocessed node-averaged timeseries data, followed by an optional dimensionality reduction step using either PCA, UMAP or autoencoders, then k-means clustering to find dynamic functional connectivity states. Place raw timeseries data into the data/raw_tseries folder, with one .csv file per subject.

Run `python3 main.py -h` for help, or see run.sh for examples. To see a demonstration, use the create_toysimulation.m script in the SimTB_model_wrapper folder to generate synthetic data, then move this to the data/raw_tseries folder and run `./run.sh`.

Python code was originally run with Python 3.6.8, using the following packages (and versions):
- numpy 1.18.5
- matplotlib 3.2.2
- tensorflow 2.2.0
- scipy 1.4.1
- sklearn 0.22.2
- umap 0.4.6
- pyclustering 0.10.1.2
- skggm 0.2.8

The SimTB wrapper runs in MATLAB and requires the SimTB toolbox (https://trendscenter.org/trends/software/simtb/). The code for the wrapper was adapted from that used in Allen et al., "Tracking whole-brain connectivity dynamics in the resting state." Cereb Cortex. 2014 24(3):663-76.

Modifications to the SimTB wrapper include:
- Generating batches of synthetic data, representing data from multiple heterogeneous subjects. Between-subject differences in HRF shape, noise parameters, and underlying state time courses.
- State FC matrices are randomly generated for a given dataset.
- Underlying state time courses are randomly sampled from a hidden Markov model.
