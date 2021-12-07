# MIT License

# Copyright (c) 2021 Arthur Spencer

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE

import os
run_parallel_preproc = True
preproc_only = False
if run_parallel_preproc:
		os.environ['OPENBLAS_NUM_THREADS'] = '1'
		import multiprocessing as proc
		n_jobs = proc.cpu_count()
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import fcCluster
import fcPreproc
import fcUtils as utils
from sklearn import metrics
from scipy import stats
import argparse
from argparse import RawTextHelpFormatter


def run_preproc(path_, datadir, window_shape, window_size, step, hcp=False):
	'''
	Run sliding window correations using fcPreproc.get_dfc
	'''

	# load raw timeseries
	subjs = fcPreproc.load_data(path_, hcp=hcp)

	nsubjs = len(subjs)
	n_nodes = len(subjs[0])
	batch_size_preproc = 256
	batch_dfc_preproc = False

	print('%i subjects' % nsubjs)
	print('%i nodes' % n_nodes)
	print('%i time points' % len(subjs[0,0]))

	# for large datasets (e.g. hundreds of HCP subjects) save in batches
	if nsubjs>batch_size_preproc:
		print('Running dfc in batches of %i' % batch_size_preproc)
		batch_dfc_preproc = True

	if run_parallel_preproc:
		# one core per subject, makes it way quicker
		p = proc.Pool(processes=n_jobs)
		if batch_dfc_preproc:
			batch_inds = np.arange(0,nsubjs,batch_size_preproc)

			for i, ind in enumerate(batch_inds):
				batch_subjs = subjs[ind:min([ind+batch_size_preproc, nsubjs])]
				print('Batch %i: %i to %i' % (i, ind, min([ind+batch_size_preproc, nsubjs])-1))
				
				subjs_arr = [(subj, window_size, window_shape, n_nodes, step, '-') for subj in batch_subjs]
				dfc_ = p.map(fcPreproc.get_dfc, subjs_arr)
				np.save(datadir + 'dfc_%i.npy'%i, np.array(dfc_))

			dfc_ = np.concatenate([np.load(datadir + 'dfc_%i.npy'%i) for i in range(len(batch_inds))], axis=0)

		else:
			subjs_arr = [(subjs[i], window_size, window_shape, n_nodes, step, '-') for i in range(nsubjs)]
			dfc_ = p.map(fcPreproc.get_dfc, subjs_arr)

	else:
		dfc_ = [None]*nsubjs
		for s in range(nsubjs):
			dfc_[s] = fcPreproc.get_dfc((subjs[s], window_size, window_shape, n_nodes, step))

	return np.array(dfc_)


def run_clustering(dfc, D, method, n_clusters=5, n_epochs=None, batch_size=None, embedded_path=None, run_elbow=False, kmax=12, show_plots=False):
	'''
	Run optional dimensionality reduction followed by clustering
	'''

	print('>>> Running clustering')
	exemplar_inds = utils.get_exemplars(dfc)

	if method=='ae':
		# Autoencoder for deep clustering
		import autoEncFC as fcae

		if embedded_path is None:
			print('Training AutoEncoder')
			autoencoder = fcae.AutoEncoder(D)
			dfc_embedded = autoencoder.encode(dfc, maximum_epoch=n_epochs, batch_size=batch_size, outdir=outdir, show_plots=show_plots)
			np.save(outdir + 'dfc_embedded.npy', dfc_embedded)
		else:
			print('loading saved embedded data from %s' % embedded_path)
			dfc_embedded = np.load(embedded_path + '/dfc_embedded.npy')

		dist = 'euclidean'

	else:
		dfc_embedded = utils.concatenate_windows(dfc)
		if method=='pca':
			from sklearn.decomposition import PCA
			print('pca to %i components' % D[-1])
			dfc_embedded = PCA(n_components=D[-1]).fit_transform(dfc_embedded)
			dist = 'euclidean'

		elif method=='umap':
			import umap
			dfc_embedded = umap.UMAP(min_dist=D[0]*1e-3, n_components=D[1], 
				n_neighbors=D[2], verbose=True).fit_transform(dfc_embedded)
			dist = 'euclidean'

		elif method=='kmeans':
			dist = 'l1'

		elif method=='kmeans-l2':
			dist = 'euclidean'

	if run_elbow:
		# run elbow criterion to find best k
		cluster_krange(dfc_embedded[exemplar_inds], kmax, dist=dist)
		clusters = None
	else:
		clusters, centroids, score = fcCluster.kmeans(dfc_embedded, exemplar_inds, n_clusters=n_clusters, dist=dist)

	return clusters, dfc_embedded


def cluster_krange(dfc, kmax, dist='euclidean'):
	'''
	Runs k-means with a range of k on the embedded data to allow plot for elbow criterion
	'''

	cvi_krange = np.zeros(kmax-1)
	for k in range(2,kmax+1):
		clusters, centroids, score = fcCluster.kmeans(dfc, np.arange(len(dfc)), n_clusters=k, dist=dist)

		labels = np.zeros(len(dfc))
		for i,clus in enumerate(clusters):
			for c in clus:
				labels[c] = i

		# cluster validity index
		states = np.array([np.mean(dfc[labels==i],axis=0) for i in range(k)])
		wcd = np.zeros(k)
		bcd = 0.
		for i in range(k):
			wcd[i] = np.sum(metrics.pairwise.euclidean_distances(dfc[labels==i],[states[i]]))

			if i<k-1:
				bcd += np.sum(metrics.pairwise.euclidean_distances(states[np.arange(i+1,k)], [states[i]]))
				
		bcd /= float(k*(k-1))/2
		cvi = np.sum(wcd)/(bcd*len(dfc))

		print('%i score = %f' % (k, cvi))
		cvi_krange[k-2] = cvi

	with open('../results/cvi.csv','a') as resfile:
		resfile.write(','.join(str(x) for x in cvi_krange))
		resfile.write('\n')


if __name__=='__main__':

	parser = argparse.ArgumentParser(description='Run dFC analysis with SWC, dimensionality reduction and clustering.', formatter_class=RawTextHelpFormatter)
	parser.add_argument('-k','--nclusters', help='Number of clusters. Default: 5', default=5, type=int)
	parser.add_argument('-c','--elbow', help='Run elbow criterion', action='store_true')
	parser.add_argument('-K','--kmax', help='Max k if running elbow criterion. Default: 12', default=12, type=int)
	parser.add_argument('-b','--batch_size', help='Batch size for autoencoder. Default: 50', default=50, type=int)
	parser.add_argument('-e','--n_epochs', help='Epochs for autoencoder training. Default: 100', default=100, type=int)
	parser.add_argument('-p','--plots', help='Show plots', action='store_true')
	parser.add_argument('-m','--model_data', help='Data generated by the SimTB model. If true, checks clustering performance against ground truth', action='store_true')
	parser.add_argument('-hcp','--hcp_data', help='HCP data (has a different file structure to SimTB data)', action='store_true')
	parser.add_argument('-t','--TR', help='Repetition time (TR) of the data, in seconds. Default: 2', default=2., type=float)
	parser.add_argument('-w','--window_shape', help='Window shape for sliding-window correlations. Default: rectangle', default='rectangle', type=str)
	parser.add_argument('-l','--window_size', help='Window size for sliding-window correlations, in TR. Default: 20', default=20, type=int)
	parser.add_argument('-s','--step', help='Step size for sliding-window correlations. Default: 1', default=1, type=int)
	parser.add_argument('-E','--embedded_path', help='Path to embedded dfc data', default=None, type=str)
	parser.add_argument('method', help='Clustering/dimensionality reduction method.\
		\n\t Options: kmeans, kmeans-l2, ae, pca, umap', type=str)
	parser.add_argument('params', help=str('Parameters. \
		\n\t Method : parameters \
		\n\t ae : d1,d2,d3 (e.g. 512,256,32)\
		\n\t umap : v*10^3,u,m (e.g. 1000,64,30)\
		\n\t pca : 1,1,p (e.g. 1,1,32)\
		\n\t kmeans or kmeans-l2 : 1,1,1 (just a place-holder).'), type=str)
	parser.add_argument('func_path', help='Path to raw data directory.', type=str)

	args = parser.parse_args()
	D = [int(d) for d in args.params.split(',')]

	if args.method not in ['kmeans', 'kmeans-l2', 'pca', 'umap', 'ae']:
		raise Exception('Method not recognised - choose from kmeans, kmeans-l2, pca, umap, ae')

	datadir = '../data/%s_%i_%s_%i/' % (args.func_path.split('/')[-2],
		args.window_size, args.window_shape, args.step)
	outdir = '../results/%s_%i_%s_%i_k%i_%s' % (args.func_path.split('/')[-2],
		args.window_size, args.window_shape, args.step, args.nclusters, args.method)

	if args.method in ['ae','umap']:
		for d in D:
			outdir += '_%i'%d
	elif args.method=='pca':
		outdir += '_%i'%D[2]
	tt = datetime.datetime.now()
	outdir += '_%i_%i_%i_%i_%i_%i/' % (tt.year, tt.month, tt.day, tt.hour, tt.minute, tt.second)

	print('Data folder: ' + datadir)
	if not os.path.isdir(datadir):
		os.mkdir(datadir)
	print('Output folder: ' + outdir)
	os.mkdir(outdir)

	if os.path.isfile(datadir + 'dfc.npy'):
		print('Loading existing dfc windows')
		dfc = np.array(np.load(datadir + 'dfc.npy'))

	else:
		print('No saved dfc found - running sliding-window correlations')

		t_start = time.time()

		# Run sliding-window correlations to convert timeseries into windowed dFC matrices
		dfc = run_preproc(args.func_path, datadir, args.window_shape, args.window_size, args.step, hcp=args.hcp_data)
		np.save(datadir + 'dfc.npy', dfc)

		print('dfc saved')
		runtime = time.time() - t_start
		print('SWC runtime: ' + str(datetime.timedelta(seconds=runtime)))

		if preproc_only:
			raise Exception('Preproc only')

	nsubjs = len(dfc)
	nwindows = len(dfc[0,:,0,0])
	n_nodes = len(dfc[0,0,:,0])	
	
	print('%i subjects' % nsubjs)
	print('%i windows per subject' % nwindows)
	print('%i nodes (%i features)' % (n_nodes, n_nodes*(n_nodes-1)/2))
	
	if args.model_data:
		# model_data flag also allows calculation of clustering accuracy from ground truth
		print('Model-generated data')

	# run optional dimensionality reduction followed by clustering
	clusters, dfc_embedded = run_clustering(dfc, D, args.method, n_clusters=args.nclusters,
		n_epochs=args.n_epochs, batch_size=args.batch_size, embedded_path=args.embedded_path,
		run_elbow=args.elbow, kmax=args.kmax, show_plots=args.plots)

	if args.elbow:
		print('Cluster validity index saved. Choose k and run without elbow flag.')

	else:
		# calculate temporal properties of dFC states
		frac_occ, dwell_time, states = utils.state_properties(clusters, dfc, args.TR, args.step)
		np.save(outdir + 'clusters_%s.npy' % args.method, clusters)
		np.save(outdir + 'fractional_occupancy_%s.npy' % args.method, frac_occ)
		np.save(outdir + 'dwell_time_%s.npy' % args.method, dwell_time)
		np.save(outdir + 'states_%s.npy' % args.method, states)

		if args.plots:
			fig = plt.figure()
			for i in range(args.nclusters):
				fig.add_subplot(1+1*args.model_data,args.nclusters,i+1)
				plt.imshow(utils.vec2conn(states[i]),cmap='jet',vmin=-1,vmax=1)

		if args.model_data:
			# assess clustering performance

			# get ground truth
			true_clusters, true_states = utils.load_truth(args.func_path, args.window_size, nwindows, args.step, args.nclusters)
			true_cluster_inds = [np.where(true_clusters==(i+1))[0] for i in range(args.nclusters)]
			frac_occ_true, dwell_time_true, true_states_derived = utils.state_properties(true_cluster_inds, dfc, args.TR, args.step)

			# compare state FC matrices to ground truth
			all_r = metrics.pairwise.cosine_similarity(states, true_states)
			all_r_max = metrics.pairwise.cosine_similarity(true_states_derived, true_states)
			maxr = np.mean(np.max(all_r, axis=1))
			maxr_max = np.mean(np.max(all_r_max, axis=1))
			centroid_mse = np.sum(np.square(states - true_states), axis=None)
			centroid_mse_max = np.sum(np.square(true_states_derived - true_states), axis=None)
			
			print('Centroid similarity:  %f  (true clustering gives:  %f)' % (maxr, maxr_max))
			print('Centroid error:       %f  (true clustering gives:  %f)' % (centroid_mse, centroid_mse_max))

			# match clusters to true states
			true_states_adj = np.zeros([args.nclusters, args.nclusters])
			rmax = np.zeros(args.nclusters)
			for i in range(args.nclusters):
				inds = np.unravel_index(np.argmax(all_r), shape=all_r.shape)
				rmax[inds[1]] = np.max(all_r)
				true_states_adj[inds] = 1
				all_r[inds[0],:] = 0
				all_r[:,inds[1]] = 0
			inds_true = np.argmax(true_states_adj, axis=1)

			true_states = true_states[inds_true]
			true_states_derived = true_states_derived[inds_true]
			frac_occ_true = frac_occ_true[:,inds_true]
			dwell_time_true = dwell_time_true[:,inds_true]

			if args.plots:
				for i in range(5):
					fig.add_subplot(2,args.nclusters,i+6)
					plt.imshow(utils.vec2conn(true_states[i]),cmap='jet',vmin=-1,vmax=1)

			derived_clusters = np.zeros(len(true_clusters))
			for i in range(args.nclusters):
				# determine which true cluster this corresponds to
				true_k = stats.mode(true_clusters[clusters[i]]).mode
				for ind in clusters[i]:
					derived_clusters[ind] = true_k

			# clustering performance metrics
			acc = float(np.count_nonzero(true_clusters==derived_clusters))/len(true_clusters)
			nmi = metrics.normalized_mutual_info_score(true_clusters, derived_clusters)
			ari = metrics.adjusted_rand_score(true_clusters, derived_clusters)

			print('ACC: %f' % acc)
			print('NMI: %f' % nmi)
			print('ARI: %f' % ari)

			if args.plots:
				plot_utils.plot_states(true_clusters, derived_clusters, nsubjs, nwindows)

			fo_mse = np.mean((frac_occ_true - frac_occ)**2, axis=None)
			dt_mse = np.mean((dwell_time_true - dwell_time)**2, axis=None)

			print('Fractional occupancy mean-squared error: %f' % fo_mse)
			print('Dwell time mean-squared error:           %f' % dt_mse)

			filename = '%s_%i_%s_%i' % (args.func_path.split('/')[-2], args.window_size, args.window_shape, args.step)
			for d in D:
				filename += '_%i' % d
			with open('../results/%s.csv' % (filename),'a') as resfile:
				resfile.write('%f,%f,%f,%f,%f,%f,%f\n'%
					(maxr, centroid_mse, acc, nmi, ari, fo_mse, dt_mse))

			if args.plots:
				plt.show()