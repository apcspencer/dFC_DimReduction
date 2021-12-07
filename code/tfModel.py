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
Autoencoder architecture for deep clustering
'''

import numpy as np
from tensorflow.keras import layers, Model

def autoencoder(D, n_nodes, batch_size=None):

	input_shape = int(n_nodes*(n_nodes-1)/2)
	input_img = layers.Input(shape=(input_shape))
	k_init = 'glorot_uniform'

	x = layers.Dense(D[0], activation='relu', kernel_initializer=k_init)(input_img)
	x = layers.Dense(D[1], activation='relu', kernel_initializer=k_init)(x)
	encoded = layers.Dense(D[2], kernel_initializer=k_init)(x)

	x = layers.Dense(D[1], activation='relu', kernel_initializer=k_init)(encoded)
	x = layers.Dense(D[0], activation='relu', kernel_initializer=k_init)(x)
	decoded = layers.Dense(input_shape, kernel_initializer=k_init)(x)

	model = Model(input_img, decoded)
	encoder = Model(input_img, encoded)

	return model, encoder