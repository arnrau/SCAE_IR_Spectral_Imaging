##########################################################
# Arne 2018
#
# MODERN MLP WITH PRETRAINED WEIGHTS FROM STACKED CONTRACTIVE AE
#
#########################################################
from __future__ import print_function

import os
import sys
import timeit

import numpy as np
np.set_printoptions(threshold=np.inf)
numpy_rng = np.random
from numpy import random as rng
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
srng = RandomStreams()
#import cPickle as pickle #Python 2.x
import _pickle as cPickle #Python 3.x
import timeit
import time
###################################################################################################################################
# Stacked Contractive Autoencoder implementation used in the Paper Raulf et al., 2019 :
#  Deep representation learning for domain adaptatable classification of infrared spectral imaging data
#
#  Core implementation is based on Theano [(The Theano Development Team);
#                                 Theano: A Python framework for fast computation of mathematical expressions]
#
#
#   Great inspiration for certain code parts, i found in the 
#   Deep Learning Tutorial (Release 0.1) from the LISA lab, University of Montreal(2015).
#   Alec Radford and François Chollets Blogs/Githubs etc.
###################################################################################################################################
# we used virtual envs of anaconda (Version 2; list of installed packages will be submitted with the code)
# Installed OS version first Ubuntu 14.04 LTS, later stage of the project elementary os 0.4.1 Loki(based on Ubuntu 16.04 LTS)
# NVIDIA 1080 TI (Driver Version: 375.66)
#   
#
#   SYNOPSIS:
#
#   python finetune.py | tee -a finetune.out 
#
#
#
#
#
#
###################################################################################################################################

# 
#!------------------------------------------------------------------------------------------------------------------------
# PRETRAIŃED WEIGHTS FROM CONTRACTIVE STACKED AUTOENCODER ARE LOADED
#!------------------------------------------------------------------------------------------------------------------------
start_import = time.time()


w1 =  np.load("parameters_CO722B_L2Norm_pretain_w1d_2000E_F450WVN.npy" ) 
b1 =  np.load("parameters_CO722B_L2Norm_pretain_b1d_2000E_F450WVN.npy" )
w2 =  np.load("parameters_CO722B_L2Norm_pretain_w2d_2000E_F450WVN.npy" )
b2 =  np.load("parameters_CO722B_L2Norm_pretain_b2d_2000E_F450WVN.npy" )
w3 =  np.load("parameters_CO722B_L2Norm_pretain_w3d_2000E_F450WVN.npy" )
b3 =  np.load("parameters_CO722B_L2Norm_pretain_b3d_2000E_F450WVN.npy" )
w4 =  np.load("parameters_CO722B_L2Norm_pretain_w4d_2000E_F450WVN.npy" )
b4 =  np.load("parameters_CO722B_L2Norm_pretain_b4d_2000E_F450WVN.npy" )
w5 =  np.load("parameters_CO722B_L2Norm_pretain_w5d_2000E_F450WVN.npy" )
b5 =  np.load("parameters_CO722B_L2Norm_pretain_b5d_2000E_F450WVN.npy" )
w6 =  np.load("parameters_CO722B_L2Norm_pretain_w6d_2000E_F450WVN.npy" )
b6 =  np.load("parameters_CO722B_L2Norm_pretain_b6d_2000E_F450WVN.npy" )



w1 = theano.shared(w1, borrow=True)
b1 = theano.shared(b1, borrow=True)
w2 = theano.shared(w2, borrow=True)
b2 = theano.shared(b2, borrow=True)
w3 = theano.shared(w3, borrow=True)
b3 = theano.shared(b3, borrow=True)
w4 = theano.shared(w4, borrow=True)
b4 = theano.shared(b4, borrow=True)
w5 = theano.shared(w5, borrow=True)
b5 = theano.shared(b5, borrow=True)
w6 = theano.shared(w6, borrow=True)
b6 = theano.shared(b6, borrow=True)


stop_import = time.time()
print('time for import pretrained weights and biases',stop_import-start_import)
# RELEVANT FUNCTIONS WITH RESPECT TO NN MODEL
def floatX(X):
	return np.asarray(X, dtype=theano.config.floatX)

def relu(X):
	return T.maximum(X, 0.)

def softmax(X):
	e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
	return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
	grads = T.grad(cost=cost, wrt=params)
	updates = []
	for p, g in zip(params, grads):
		acc = theano.shared(p.get_value() * 0.)
		acc_new = rho * acc + (1 - rho) * g ** 2
		gradient_scaling = T.sqrt(acc_new + epsilon)
		g = g / gradient_scaling
		updates.append([acc, acc_new])
		updates.append([p, p - lr * g])
	return updates

def dropout(X, p=0.):
	if p > 0:
		retain_prob = 1 - p
		X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
		#X *= numpy_rng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
		X /= retain_prob
	return X

def _init_hidden_weights(n_in, n_out):
	rng = np.random.RandomState(1234)
	weights = np.asarray(
		rng.uniform(
			low=-np.sqrt(6. / (n_in + n_out)),
			high=np.sqrt(6. / (n_in + n_out)),
			size=(n_in, n_out)
		),
		dtype=theano.config.floatX
	)
	bias = np.zeros((n_out,), dtype=theano.config.floatX)
	return ( 
		theano.shared(value=weights, name='W', borrow=True),
		theano.shared(value=bias, name='b', borrow=True) 
		)
#!------------------------------------------------------------------------------------------------------------------------
# MODEL
#!------------------------------------------------------------------------------------------------------------------------

def model(X, w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, w6, b6,w_o, b_o, p_drop_input,p_drop_input2,p_drop_input3,p_drop_input4, p_drop_hidden):
	
	X = dropout(X, p_drop_input)
	
	h = relu(T.dot(X, w1)+b1)
	
	h = dropout(h, p_drop_input2)
	
	h2 = relu(T.dot(h, w2)+b2)
	h2 = dropout(h2, p_drop_input3)

	h3 = dropout(h2, p_drop_input3)
	h3 = relu(T.dot(h2, w3)+b3)	
	

	h4 = dropout(h3, p_drop_hidden)
	h4 = relu(T.dot(h3, w4)+b4)	


	h5 = dropout(h4, p_drop_hidden)
	h5 = relu(T.dot(h4, w5)+b5)

	h6 = dropout(h5, p_drop_hidden)
	h6 = relu(T.dot(h5, w6)+b6)	

	py_x = softmax(T.dot(h6, w_o)+b_o)
	
	return h, h2, h3, h4, h5, h6, py_x

#!------------------------------------------------------------------------------------------------------------------------
# FETCH AND PREPARE DAT
#
#!------------------------------------------------------------------------------------------------------------------------
start_import2 = time.time()
dat = np.load('finetune_data.npy')

dat = dat[:,:,:450]
x,y,z  = dat.shape

cdat = dat.reshape(x*y,z, order='C')
cdat = np.array( cdat ,dtype='float32')
train = np.load('finetune_y_ONEHOT.npy')

x2,y2,z2  = train.shape
train = train.reshape(x2*y2,z2, order='C')

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(cdat, train, test_size=0.4,random_state=654)# hist: 42,34,54,347,654



from sklearn.metrics import recall_score
from sklearn.preprocessing import normalize

trX = normalize(x_train, axis=1, norm='l2')
trY = y_train

teX = normalize(x_test, axis=1, norm='l2')
teY = y_test


stop_import2 = time.time()
print('TIME FOR IMPORT DATA',stop_import2 - start_import2)





#!------------------------------------------------------------------------------------------------------------------------
# SYMBOLIC VARIABLES DECLARED AND PROCESSED IN THEANO
#!------------------------------------------------------------------------------------------------------------------------
X = T.fmatrix()
#Y = T.fmatrix()
Y = T.imatrix()

w_o, b_o = _init_hidden_weights(100, 19)

noise_h, noise_h2, noise_h3, noise_h4, noise_h5, noise_h6, noise_py_x = model(X,w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, w6, b6,w_o, b_o, 0.,0., 0., 0.3, 0.5)
h, h2, h3, h4, h5, h6, py_x = model(X, w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, w6, b6, w_o, b_o, 0., 0., 0., 0.,0.)

y_x = T.argmax(py_x, axis=1)

cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))

params = [w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, w6, b6,w_o, b_o]

updates = RMSprop(cost, params, lr=0.001)

print('Compile Train Function')
print(theano.config.blas.ldflags)

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)#, mode='DebugMode')

print('Compile Prediction Function')

predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)#,mode='DebugMode')

print('Compile Function Pvalues of Prediction')


#BATCH TRAINING FOR UPDATING THE WEIGHTS
for i in range(15000):
	for start, end in zip(range(0, len(trX), 500), range(500, len(trX), 500)):
		cost = train(trX[start:end], trY[start:end])
	
	print(i,"Ep",'Tr_Cost', cost,"Tes_Acc",np.mean(np.argmax(teY, axis=1) == predict(teX)))


#!------------------------------------------------------------------------------------------------------------------------
# PARAMETER SAVING FROM MLP 
#!------------------------------------------------------------------------------------------------------------------------
np.save("parameters_pretain_w1d_2000E_F450WVN_FINETUNE_1.npy"  ,w1.get_value(True) )
np.save("parameters_pretain_b1d_2000E_F450WVN_FINETUNE_1.npy"  ,b1.get_value(True) )
np.save("parameters_pretain_w2d_2000E_F450WVN_FINETUNE_1.npy"  ,w2.get_value(True) )
np.save("parameters_pretain_b2d_2000E_F450WVN_FINETUNE_1.npy"  ,b2.get_value(True) )
np.save("parameters_pretain_w3d_2000E_F450WVN_FINETUNE_1.npy"  ,w3.get_value(True) )
np.save("parameters_pretain_b3d_2000E_F450WVN_FINETUNE_1.npy"  ,b3.get_value(True) )
np.save("parameters_pretain_w4d_2000E_F450WVN_FINETUNE_1.npy"  ,w4.get_value(True) )
np.save("parameters_pretain_b4d_2000E_F450WVN_FINETUNE_1.npy"  ,b4.get_value(True) )
np.save("parameters_pretain_w5d_2000E_F450WVN_FINETUNE_1.npy"  ,w5.get_value(True) )
np.save("parameters_pretain_b5d_2000E_F450WVN_FINETUNE_1.npy"  ,b5.get_value(True) )
np.save("parameters_pretain_w6d_2000E_F450WVN_FINETUNE_1.npy"  ,w6.get_value(True) )
np.save("parameters_pretain_b6d_2000E_F450WVN_FINETUNE_1.npy"  ,b6.get_value(True) )
np.save("parameters_pretain_w_o_2000E_F450WVN_FINETUNE_1.npy"  ,w_o.get_value(True) )
np.save("parameters_pretain_b_o_2000E_F450WVN_FINETUNE_1.npy"  ,b_o.get_value(True) )


from sklearn.metrics import confusion_matrix

pre = np.asarray(predict(teX))
tru = np.argmax(teY, axis=1)
cm1 = confusion_matrix(tru,pre)
print(cm1)
np.save("confmat_Finetune_TestSet.npy", cm1)
np.save("pred_Finetune_TestSet.npy", pre)