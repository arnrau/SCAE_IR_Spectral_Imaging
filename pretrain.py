from __future__ import print_function

import os
import sys
import timeit

import numpy as np
np.set_printoptions(threshold=np.inf)
numpy_rng = np.random
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from sklearn.preprocessing import normalize
#import cPickle as pickle
#depends on Python 2|3 
import _pickle as cPickle

import timeit
import time
###################################################################################################################################
# Stacked Contractive Autoencoder implementation used in the Paper Raulf et al. :
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
#   python pretrain.py | tee -a pretrain.out 
#
#
#
#
#
#
###################################################################################################################################
start_import = time.time()
def impo_dat(datname):
    dat = np.load(datname)
    dat = dat[:,:,:450]
    x,y,z  = dat.shape
    cdat = dat.reshape(x*y,z)
    cdat = np.array( cdat ,dtype='float32')
    print(cdat.shape)
    return cdat

cdat = impo_dat('pretrain_data.npy')
cdat_scale = normalize(cdat, axis=1, norm='l2')
train_set_x = theano.shared(np.asarray(cdat_scale, dtype=theano.config.floatX) )
stop_import = time.time()
print('time for import files',stop_import - start_import)

###############################################################################
###############################################################################
###############################################################################


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        self.input = input
        
        if W is None:
            W_values = np.asarray(
                numpy_rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]
        #self.params = [self.W, self.b, self.output]

class AE(object):
    def __init__(self, numpy_rng, theano_rng=None, input=None, n_visible=450, n_hidden=450, W=None, bhid=None, bvis=None, z_out=None):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        #self.z_out = z_out
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if not W:
            initial_W = np.asarray(
                numpy_rng.uniform(
                    low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * np.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                    ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)
            

        if not bvis:
            bvis = theano.shared(
                value=np.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                borrow=True)
        if not bhid:
            bhid = theano.shared(
                value=np.zeros(n_hidden,dtype=theano.config.floatX ),
                name='b', borrow=True)

        self.W = W
        self.b = bhid
        self.b_prime = bvis
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        self.z_out = theano.shared(  [T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)] , dtype='float32')
        
        if input is None:
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.params =  [self.W, self.b, self.b_prime] 

    def get_hidden_values(self, input):
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_updates(self, learning_rate, lam=1e-3):
        tilde_x = self.x
        y = self.get_hidden_values(tilde_x)
        #z, z_out = self.get_reconstructed_input(y)
        z = self.get_reconstructed_input(y)
        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
  
        mse = T.mean((z - self.x) ** 2)
        lam = 1e-3        
        dh = z * (1 - z)
        contractive = lam * T.sum(dh**2 * T.sum(self.W**2, axis=1))
        cost = mse + contractive       

        gparams = T.grad(cost, self.params)
   
        updates = [(param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)]

        return (cost, updates,[z])





class SA(object):
    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        n_ins=450,
        hidden_layers_sizes=[450,900, 450, 100, 100, 100,10,100, 100, 100, 10, 100, 100, 50, 50, 50, 50, 50, 3,10],
        n_outs=10,
        z=None
    ):


        self.sigmoid_layers = []
        self.AE_layers = []
        #self.AE_layers_out = []
        self.params = []
        self.hidden = []
        self.n_layers = len(hidden_layers_sizes) 
        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        
        self.x = T.matrix('x')  
        self.y = T.ivector('y') 
                                

        self.z = T.matrix('z')

        for i in range(self.n_layers):

            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output



            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid)

            print("hidden_layers_sizes",hidden_layers_sizes)
            print("self.n_layers",self.n_layers)

            self.sigmoid_layers.append(sigmoid_layer)

            self.params.extend(sigmoid_layer.params)

            AE_layer = AE(numpy_rng=numpy_rng,
                        theano_rng=theano_rng,
                        input=layer_input,
                        n_visible=input_size,
                        n_hidden=hidden_layers_sizes[i],
                        W=sigmoid_layer.W,
                        bhid=sigmoid_layer.b)
            self.AE_layers.append(AE_layer)


    def pretraining_functions(self, train_set_x, batch_size):
        batch_size = 1
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
        
        index = T.lscalar('index')  
        
        learning_rate = T.scalar('lr')  
        
        batch_begin = index * batch_size
        
        batch_end = batch_begin + batch_size
        
        pretrain_fns = []
        z_outs = []

        for ae in self.AE_layers:

            cost,updates,z = ae.get_cost_updates(learning_rate)
            fn = theano.function( inputs=[
                index, 
                theano.In(learning_rate, value=0.1)
                ],
            outputs=cost,
            updates=updates, 
            givens = {self.x: train_set_x[batch_begin: batch_end]})
            pretrain_fns.append(fn)





            z_out = ae.get_reconstructed_input( self.sigmoid_layers[-1].output )
            fn2 = theano.function( inputs=[self.sigmoid_layers[-1].output],
                outputs=z_out,
                on_unused_input='ignore',
                givens = {self.x: train_set_x[batch_begin: batch_end]}
                )
            z_outs.append(fn2)


        return pretrain_fns, z_outs


    def save_params(self, filename):
        output = open(filename, 'wb')
        for p in range(len(self.params)):
            pickle.dump(self.params[p].get_value(), output)
        output.close()

    def compute_out(self, input):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])


    def save_hidden(self, filename):
        out = open(filename, 'wb')
        for p in range(len(self.sigmoid_layers.output)):
            pickle.dump(self.sigmoid_layers.output[p].eval(), out)
        out.close()


def test_SCA(pretraining_epochs=2000,
             pretrain_lr=0.003,
             dataset=input, 
             batch_size=500):



    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    print("n train batches!", n_train_batches)
    print("batches size!",  batch_size)
    numpy_rng = np.random.RandomState(89677)
    print('... building the model')
    #####################
    # CLass contruction
    #####################
    sda = SA(
        numpy_rng=numpy_rng,
        n_ins=450,
        hidden_layers_sizes=[450,900, 450, 100, 100, 100,10,100, 100, 100, 10, 100, 100, 50, 50, 50, 50, 50, 3,10])
    

    print('... getting the pretraining functions')
    
    pretraining_fns, z_outs = sda.pretraining_functions(train_set_x=train_set_x, batch_size=batch_size)

    print('... pre-training the model')
    start_time = timeit.default_timer()
    ## Pretrain layer-wise
    
    for i in range(sda.n_layers):
        for epoch in range(pretraining_epochs):
            c1 = []
            for batch_index in range(n_train_batches):
                c1.append( pretraining_fns[i](index=batch_index, lr=pretrain_lr) )
            print('Pre-training layer %i, epoch %d, cost %f' % (i, epoch, np.mean(c1, dtype='float32')))

    

    print(sda.params) 
    np.save("parameters_CO722B_L2Norm_pretain_w1d_2000E_F450WVN.npy" ,sda.params[0].get_value(borrow=True))
    np.save("parameters_CO722B_L2Norm_pretain_b1d_2000E_F450WVN.npy" ,sda.params[1].get_value(borrow=True))
    np.save("parameters_CO722B_L2Norm_pretain_w2d_2000E_F450WVN.npy" ,sda.params[2].get_value(borrow=True))
    np.save("parameters_CO722B_L2Norm_pretain_b2d_2000E_F450WVN.npy" ,sda.params[3].get_value(borrow=True))
    np.save("parameters_CO722B_L2Norm_pretain_w3d_2000E_F450WVN.npy" ,sda.params[4].get_value(borrow=True))
    np.save("parameters_CO722B_L2Norm_pretain_b3d_2000E_F450WVN.npy" ,sda.params[5].get_value(borrow=True))
    np.save("parameters_CO722B_L2Norm_pretain_w4d_2000E_F450WVN.npy" ,sda.params[6].get_value(borrow=True))
    np.save("parameters_CO722B_L2Norm_pretain_b4d_2000E_F450WVN.npy" ,sda.params[7].get_value(borrow=True))
    np.save("parameters_CO722B_L2Norm_pretain_w5d_2000E_F450WVN.npy" ,sda.params[8].get_value(borrow=True))
    np.save("parameters_CO722B_L2Norm_pretain_b5d_2000E_F450WVN.npy" ,sda.params[9].get_value(borrow=True))
    np.save("parameters_CO722B_L2Norm_pretain_w6d_2000E_F450WVN.npy" ,sda.params[10].get_value(borrow=True))
    np.save("parameters_CO722B_L2Norm_pretain_b6d_2000E_F450WVN.npy" ,sda.params[11].get_value(borrow=True))
    np.save("parameters_CO722B_L2Norm_pretain_w7d_2000E_F450WVN.npy" ,sda.params[12].get_value(borrow=True))
    np.save("parameters_CO722B_L2Norm_pretain_b7d_2000E_F450WVN.npy" ,sda.params[13].get_value(borrow=True))
    np.save("parameters_CO722B_L2Norm_pretain_w8d_2000E_F450WVN.npy" ,sda.params[14].get_value(borrow=True))
    np.save("parameters_CO722B_L2Norm_pretain_b8d_2000E_F450WVN.npy" ,sda.params[15].get_value(borrow=True))
    np.save("parameters_CO722B_L2Norm_pretain_w9d_2000E_F450WVN.npy" ,sda.params[16].get_value(borrow=True))
    np.save("parameters_CO722B_L2Norm_pretain_b9d_2000E_F450WVN.npy" ,sda.params[17].get_value(borrow=True))
    np.save("parameters_CO722B_L2Norm_pretain_w10d_2000E_F450WVN.npy",sda.params[18].get_value(borrow=True))
    np.save("parameters_CO722B_L2Norm_pretain_b10d_2000E_F450WVN.npy",sda.params[19].get_value(borrow=True))
    np.save("parameters_CO722B_L2Norm_pretain_w11d_2000E_F450WVN.npy",sda.params[20].get_value(borrow=True))
    np.save("parameters_CO722B_L2Norm_pretain_b11d_2000E_F450WVN.npy",sda.params[21].get_value(borrow=True))
    np.save("parameters_CO722B_L2Norm_pretain_w12d_2000E_F450WVN.npy",sda.params[22].get_value(borrow=True))
    np.save("parameters_CO722B_L2Norm_pretain_b12d_2000E_F450WVN.npy",sda.params[23].get_value(borrow=True))
    np.save("parameters_CO722B_L2Norm_pretain_w13d_2000E_F450WVN.npy",sda.params[24].get_value(borrow=True))
    np.save("parameters_CO722B_L2Norm_pretain_b13d_2000E_F450WVN.npy",sda.params[25].get_value(borrow=True))
    np.save("parameters_CO722B_L2Norm_pretain_w14d_2000E_F450WVN.npy",sda.params[26].get_value(borrow=True))
    np.save("parameters_CO722B_L2Norm_pretain_b14d_2000E_F450WVN.npy",sda.params[27].get_value(borrow=True))
    np.save("parameters_CO722B_L2Norm_pretain_w15d_2000E_F450WVN.npy",sda.params[28].get_value(borrow=True))
    np.save("parameters_CO722B_L2Norm_pretain_b15d_2000E_F450WVN.npy",sda.params[29].get_value(borrow=True))
    np.save("parameters_CO722B_L2Norm_pretain_w16d_2000E_F450WVN.npy",sda.params[30].get_value(borrow=True))
    np.save("parameters_CO722B_L2Norm_pretain_b16d_2000E_F450WVN.npy",sda.params[31].get_value(borrow=True))
    np.save("parameters_CO722B_L2Norm_pretain_w17d_2000E_F450WVN.npy",sda.params[32].get_value(borrow=True))
    np.save("parameters_CO722B_L2Norm_pretain_b17d_2000E_F450WVN.npy",sda.params[33].get_value(borrow=True))
    np.save("parameters_CO722B_L2Norm_pretain_w18d_2000E_F450WVN.npy",sda.params[34].get_value(borrow=True))
    np.save("parameters_CO722B_L2Norm_pretain_b18d_2000E_F450WVN.npy",sda.params[35].get_value(borrow=True))
    np.save("parameters_CO722B_L2Norm_pretain_w19d_2000E_F450WVN.npy",sda.params[36].get_value(borrow=True))
    np.save("parameters_CO722B_L2Norm_pretain_b19d_2000E_F450WVN.npy",sda.params[37].get_value(borrow=True))
    np.save("parameters_CO722B_L2Norm_pretain_w20d_2000E_F450WVN.npy",sda.params[38].get_value(borrow=True))
    np.save("parameters_CO722B_L2Norm_pretain_b20d_2000E_F450WVN.npy",sda.params[39].get_value(borrow=True))




    end_time = timeit.default_timer()


print(end_time-start_import, "pretrain in sec.")
test_SCA()