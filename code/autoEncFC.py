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

'''
Autoencoder class script for deep clustering. Autoencoder architecture is
defined in tfModel.py
'''

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
from matplotlib import cm
import tensorflow as tf
from fcUtils import *
import tfModel


class AutoEncoder():

	def __init__(self, D):

		self.D = D
		self.dfc_mean = None
		self.dfc_std = None

		print('Autoencoder. Layers: ')
		print(D)


	def encode(self, dfc, maximum_epoch=100, batch_size=50, outdir=None, l=0.001, show_plots=False):
		'''
		Trains autoencoder with the architecture defined in tfModel.py, and returns the 
		encoded data for all dFC windows.
		'''

		dfc = dfc.astype(np.float32)

		n_nodes = len(dfc[0,0])
		dfc = np.concatenate(dfc, axis=0)
		nwindows = len(dfc)

		# normalise data
		self.dfc_mean =  np.mean(dfc, axis=0)
		self.dfc_std = np.std(dfc, axis=0)
		self.dfc_std = np.max([self.dfc_std,0.01*np.ones([n_nodes, n_nodes])], axis=0) # to prevent divide-by-zeros
		dfc -= self.dfc_mean
		dfc /= self.dfc_std

		print('Training autoencoder')

		train_data = np.array([conn2vec(dfc[i]) for i in range(nwindows)]).astype(np.float32)
		model, encoder = tfModel.autoencoder(self.D, n_nodes, batch_size=batch_size)

		cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=outdir, save_weights_only=True, verbose=0)

		# Adam optimiser to minimise mean-squared error between output and input
		model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=l, epsilon=1e-07, amsgrad=False), loss='mse')
		model.build(train_data.shape)
		print(model.summary())

		# save training history for optional plotting
		history = model.fit(train_data, train_data, batch_size=batch_size, epochs=maximum_epoch,
			shuffle=True, verbose=1, callbacks=[cp_callback])
		print('Training Done')

		if show_plots:
			fig = plt.figure()
			plt.plot(np.log(history.history['loss']))
			plt.title('Loss')
			plt.ylabel('log loss')
			plt.xlabel('epoch')

		if outdir is not None:
			np.save(outdir + 'loss.npy', np.array(history.history['loss']))

		test = model.predict(train_data, batch_size=batch_size)
		encoded_imgs = encoder.predict(train_data, batch_size=batch_size)
		print('Encoding Done')

		if show_plots:
			test = [vec2conn(im) for im in test]
			self.encoder_plots(dfc, encoded_imgs, test)

		return encoded_imgs


	def encoder_plots(self, dfc, encoded_imgs, test):
		'''
		plot autoencoder demo
		'''

		fig, axes = plt.subplots(nrows=5, ncols=4)
		cols = ['True', 'Encoded', 'Decoded', 'Error']

		for ax, col in zip(axes[0], cols):
			ax.set_title(col)

		for i in range(5):
			ind = int(i*np.floor(len(dfc)/5))
			axes[i,0].imshow(dfc[ind]*self.dfc_std + self.dfc_mean, cmap='jet',vmin=-1,vmax=1)

			axes[i,1].imshow(np.reshape(encoded_imgs[ind],[encoded_imgs.shape[1],1]), cmap='jet',vmin=-1,vmax=1)
			axes[i,1].get_xaxis().set_visible(False)
			axes[i,1].get_yaxis().set_visible(False)

			axes[i,2].imshow(test[ind]*self.dfc_std + self.dfc_mean, cmap='jet',vmin=-1,vmax=1)

			axes[i,3].imshow((test[ind]-dfc[ind])*self.dfc_std, cmap='jet',vmin=-0.1,vmax=0.1)
