#!/bin/bash

python3 main.py -m ae 512,256,32 ../data/raw_tseries/synthetic_tseries/
#python3 main.py -m umap 1000,64,30 ../data/raw_tseries/synthetic_tseries/
#python3 main.py -m pca 1,1,32 ../data/raw_tseries/synthetic_tseries/
#python3 main.py -m kmeans 1,1,1 ../data/raw_tseries/synthetic_tseries/
#python3 main.py -m kmeans-l2 1,1,1 ../data/raw_tseries/synthetic_tseries/

#python3 main.py -hcp -k 4 -t 0.720 -w 55 -s 2 ae 1024,256,64 ../data/HCP/hcp_tseries/
