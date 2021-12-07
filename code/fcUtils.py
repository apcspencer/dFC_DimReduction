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

import numpy as np
from scipy import signal
import fcPreproc


def conn2vec(conn):
	'''
	Concatenate edges in connectivity matrix so the upper triangle of the
	NxN matrix is represented by a single N(N-1)/2 element feature vector
	'''
	n_nodes = len(conn)
	vec = np.zeros(int(n_nodes*(n_nodes-1)/2))
	upper = 0
	for i in range(n_nodes-1):
		lower = upper
		upper = lower+(n_nodes-1)-i
		vec[lower:upper] = conn[i,i+1::]

	return vec


def vec2conn(vec):
	'''
	Perform the opposite of conn2vec; convert a N(N-1)/2 element feature
	vector into a NxN connectome
	'''
	n_nodes = int(np.ceil(np.sqrt(len(vec)*2)))
	conn = np.zeros([n_nodes, n_nodes])
	upper = 0
	for i in range(n_nodes-1):
		lower = upper
		upper = lower+(n_nodes-1)-i
		conn[i, i+1:] = vec[lower:upper]

	# copy upper to lower triangle
	i_lower = np.tril_indices(n_nodes, -1)
	conn[i_lower] = conn.T[i_lower]

	return conn


def concatenate_windows(dfc):
	'''
	Convert all dfc windows from all subjects into matrix
	dfc [n subjects x n windows x n nodes x n nodes]
	returns N x D matrix where N = subjects*windows; D = nodes*(nodes-1)/2
	'''
	nsubjs = len(dfc)
	nwindows = len(dfc[0,:,0,0])
	n_nodes = len(dfc[0,0,:,0])

	all_windows = np.zeros([nsubjs*nwindows, int(n_nodes*(n_nodes-1)/2)])

	for i in range(nsubjs):
		for w in range(nwindows):
			all_windows[i*nwindows+w, :] = conn2vec(dfc[i,w,:,:])

	return all_windows


def get_exemplars(dfc):
	'''
	Get high-variance FC windows to initialise k-means clustering
	'''

	nsubjs = len(dfc)
	nwindows = len(dfc[0,:,0,0])
	n_nodes = len(dfc[0,0,:,0])
	exemplar_inds = []
	n_per_subj = np.zeros(nsubjs)

	for i in range(nsubjs):
		window_var = np.zeros(nwindows)
		for j in range(nwindows):
			window_var[j] = np.var(conn2vec(dfc[i,j]))

		max_inds = signal.argrelextrema(window_var, np.greater)[0]
		exemplar_inds.append(np.array(max_inds) + i*nwindows)

		n_per_subj[i] = len(max_inds)

	exemplar_inds = np.concatenate(exemplar_inds, axis=0)

	print('Using exemplars for kmeans init:')
	print('   original trace %i windows per subject' % nwindows)
	print('   median %i (range %i-%i) exemplars per subject' % (np.median(n_per_subj), min(n_per_subj), max(n_per_subj)))
	print('   total %i exemplars' % len(exemplar_inds))

	return exemplar_inds


def load_truth(func_path, window_size, nw, step_size, nclusters):
	'''
	load ground truth of SimTB synthetic data
	'''

	# load true state FC matrices
	true_states = np.array([conn2vec(np.loadtxt(func_path + 'states/state%i.csv'%(i+1),
		delimiter=',')) for i in range(nclusters)])

	# load true state time course
	true_tseries = fcPreproc.load_data(func_path + 'ground_truth/', zscore=False)
	
	ns = len(true_tseries)
	true_clusters = np.zeros([ns, nw])
	for w in range(nw):
		true_clusters[:,w] = true_tseries[:,w*step_size+int(window_size/2)]
	true_clusters = np.concatenate(true_clusters,axis=0)

	return true_clusters, true_states


def state_properties(clusters, dfc, TR, step_size):
	'''
	get temporal properties of dFC states, including dwell time, fractional occupancy
	and state FC matrices
	'''

	nsubjs = dfc.shape[0]
	nwindows = dfc.shape[1]
	n_nodes = dfc.shape[2]
	nclusters = len(clusters)

	subj_states = np.zeros(nsubjs*nwindows)
	for i in range(nclusters):
		for ind in clusters[i]:
			subj_states[ind] = i
	subj_states = np.reshape(subj_states, [nsubjs, nwindows]).astype(np.int)

	frac_occ = np.zeros([nsubjs, nclusters])
	for i in range(nclusters):
		frac_occ[:,i] = np.count_nonzero(subj_states==i, axis=1)/nwindows

	dwell_time = np.zeros([nsubjs, nclusters])
	for i in range(nsubjs):
		state_count = np.zeros(nclusters)
		t = 0
		t_start = 0
		this_state = subj_states[i,0]
		while t <= nwindows:
			if t==nwindows or subj_states[i,t]!=this_state:

				dwell_time[i,this_state] += (t - t_start)*TR*step_size
				state_count[this_state] += 1

				if t<nwindows:
					t_start = t
					this_state = subj_states[i,t_start]

			t += 1

		dwell_time[i] = np.divide(dwell_time[i], state_count,
			out=np.zeros_like(dwell_time[i]), where=state_count!=0)

	all_windows = np.concatenate(dfc, axis=0)
	states = np.array([conn2vec(np.median(all_windows[c], axis=0)) for c in clusters])

	return frac_occ, dwell_time, states
