import numpy as np
import matplotlib.pyplot
import pickle

from im2col import *
from layers import *
from functions import *

from collections import OrderedDict

class SimpleConvNet:
    def __init__(self, input_dim=(3,80,80),
                conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},
                hidden_size=100,
                output_size=2,
                weight_init_std=0.01):
        filter_num1 = conv_param['filter_num']
        filter_size1 = conv_param['filter_size']
        filter_pad1 = conv_param['pad']
        filter_stride1 = conv_param['stride']
        
        """
        #parameters for convolution2, (W1_1, b1_1)
        filter_num2 = int(filter_num1 / 10)
        filter_size2 = filter_size1 +4
        filter_pad2 = filter_pad1
        filter_stride2 = filter_stride1+5
        """
        
        input_size1 = input_dim[1]
        conv_output_size1 = (input_size1 - filter_size1 + 2*filter_pad1)/filter_stride1 + 1
        pool_output_size1 = int(filter_num1 * (conv_output_size1/2) * (conv_output_size1/2))
        #print(conv_output_size1, pool_output_size1)
        """
        conv_output_size2 = (pool_output_size1 - filter_size2 + 2*filter_pad2)/filter_stride2 + 1
        pool_output_size2 = int(filter_num2 * (conv_output_size2/4) * (conv_output_size2/4))
        print(conv_output_size2, pool_output_size2)
        """
        self.params = {}
        self.params['W1'] = weight_init_std * \
                            np.random.randn(filter_num1, input_dim[0], filter_size1, filter_size1)
        self.params['b1'] = np.zeros(filter_num1)
        """
        #append a convolution layer for DogTrainer
        self.params['W1_1'] = weight_init_std * \
                            np.random.randn(filter_num2, input_dim[0], filter_size2, filter_size2)
        self.params['b1_1'] = np.zeros(filter_num2)
        # above one is the layer appended
        """
        self.params['W2'] = weight_init_std * \
                            np.random.randn(pool_output_size1, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)
        
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'],
                                          self.params['b1'],
                                          conv_param['stride'],
                                          conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        """
        self.layers['Conv2'] = Convolution(self.params['W1_1'],
                                          self.params['b1_1'],
                                          conv_param['stride'],
                                          conv_param['pad'])
        self.layers['Relu2'] = Relu()
        self.layers['Pool2'] = Pooling(pool_h=4, pool_w=4, stride=4)
        """
        self.layers['Affine1'] = Affine(self.params['W2'],
                                        self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'],
                                        self.params['b3'])
        self.last_layer = SoftMaxWithLoss()
        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return(x)
        
    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y,t)
    
    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        acc = 0.0
        
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt) 
        
        return acc / x.shape[0]
    
    def gradient(self, x, t):
        # forward
        self.loss(x, t)
        
        # backward
        dout = 1
        dout = self.last_layer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        
        # config
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        #grads['W1_1'], grads['b1_1'] = self.layers['Conv2'].dW, self.layers['Conv2'].db
        grads['W2'], grads['b2'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W3'], grads['b3'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads
        
    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, key in enumerate(['Conv1', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i+1)]
            self.layers[key].b = self.params['b' + str(i+1)]
        
        
        