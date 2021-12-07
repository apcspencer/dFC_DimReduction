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
# SOFTWARE.

'''
Preprocessing script for dFC analysis. Runs sliding-window correlations (SWC), 
calculating functional connectivity by estimating covariance from the precision 
matrix, regularised with the L1-norm (using the inverse_covariance package).
'''

import os
import numpy as np
import multiprocessing as proc
from scipy import stats
from scipy import linalg
import inverse_covariance


def get_lambda(windowed_func):
	'''
	get the subject-specific regularisation parameter, lambda,
	using leave-n-out cross validation
	'''

	nwindows = len(windowed_func)
	n_nodes = len(windowed_func[0])
	window_size = len(windowed_func[0,0])
	
	leave_out = 5
	inds = np.arange(0,nwindows-leave_out,leave_out)

	model = inverse_covariance.QuicGraphicalLasso(lam=0., max_iter=100)

	alpha_u = 1.
	alpha_l = 0.
	alpha_rounds = 5

	for s in range(alpha_rounds):

		alpha_step = 10**-(s+1)
		alphas = np.arange(alpha_l,alpha_u,alpha_step)
		sum_ll = -np.inf*np.ones(len(alphas))

		for a in range(len(alphas)):
			model.set_params(lam=alphas[a])

			log_likelihood = 0
			for i in range(len(inds)):
				Xtrain = np.concatenate([windowed_func[:inds[i]], windowed_func[inds[i]+leave_out:]],axis=0)
				Xtrain = np.reshape(Xtrain,[n_nodes,(nwindows-leave_out)*window_size])

				model.fit(Xtrain.T)

				Xtest = windowed_func[inds[i]:inds[i]+leave_out]
				for t in range(leave_out):
					log_likelihood += model.score(Xtest[t].T)

			sum_ll[a] = log_likelihood

			if a!=0 and sum_ll[a]<=sum_ll[a-1]:
				break

		best_alpha_ind = np.argmax(sum_ll)
		alpha_l = alphas[max([best_alpha_ind-1,0])]
		alpha_u = alphas[min([best_alpha_ind+1,len(sum_ll)-1])]

	best_alpha = alphas[best_alpha_ind]

	return best_alpha


def get_fc(func, lambda_):
	'''
	Estimate functional connectivity from each BOLD timeseries window, using the
	subject-specific regularisation parameter, lambda. 
	'''

	model = inverse_covariance.QuicGraphicalLasso(lam=lambda_, max_iter=100)
	model.fit(func.T)

	cov = np.array(model.covariance_)
	D = np.sqrt(np.diag(np.diag(cov)))
	DInv = np.linalg.inv(D);
	fc = np.matmul(DInv,np.matmul(cov,DInv))
	np.fill_diagonal(fc, 0)

	return np.arctanh(fc)


def get_dfc(in_):
	'''
	run SWC for a single subject. Window the BOLD timeseries data then get lambda 
	and estimate the FC matrix for each window.
	'''

	func = in_[0]
	window_size = in_[1]
	window_shape = in_[2]
	n_nodes = in_[3]
	step = in_[4]

	n_nodes = len(func[:,0])
	if window_shape=='rectangle':
		window = np.ones(window_size)
	elif window_shape=='hamming':
		window = np.hamming(window_size)
	elif window_shape=='hanning':
		window = np.hanning(window_size)
	else:
		raise Exception('%s window shape not recognised. Choose rectangle, hamming or hanning.'%window_shape)

	inds = range(0,len(func[0])-window_size,step)
	nwindows = len(inds)
	dfc = np.zeros([nwindows,n_nodes,n_nodes])
	windowed_func = np.zeros([nwindows,n_nodes,window_size])

	for i in range(nwindows):
		this_sec = func[:,inds[i]:inds[i]+window_size]
		windowed_func[i] = this_sec*window

	lambda_ = get_lambda(windowed_func)

	for i in range(nwindows):
		dfc[i,:,:] = get_fc(windowed_func[i], lambda_)

	return dfc


def load_data(func_path, zscore=True, hcp=False):
	'''
	Load raw timeseries data.
	SimTB data is in the form of one .csv file per subject, with one node per row
	and one timepoint per column (i.e. each separated by a comma). HCP data is in
	the form of one .txt file per subject, with one node per column (each separated
	by a space) and one timepoint per row. For other data, use the same file
	structure as SimTB and run without the -hcp flag, or use the same file structure
	as HCP and run with the -hcp flag.
	'''

	files = os.listdir(func_path)
	
	if hcp:
		files = sorted([file for file in files if file.endswith('.txt')])
		subjs = np.array([stats.zscore(np.loadtxt('%s/%s' % (func_path, file)).T, axis=1) for file in files])

	else:
		files = sorted([file for file in files if file.endswith('.csv')])
		if zscore:
			subjs = np.array([stats.zscore(np.loadtxt('%s/%s' % (func_path, file), delimiter=','), axis=1) for file in files])
		else:
			# don't z-score when loading ground-truth state time courses for SimTB data
			subjs = np.array([np.loadtxt('%s/%s' % (func_path, file), delimiter=',') for file in files])

	return subjs
