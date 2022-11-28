# coding: utf-8
import sys, os
sys.path.append(os.path.abspath('~/deep/deep-learning-from-scratch/'))
sys.path.append(os.path.abspath('src/'))
import numpy as np
import matplotlib.pyplot as plt
from mnist import load_mnist
from simpleconvnet import SimpleConvNet
from trainer import Trainer
from collections import OrderedDict
from LoadDogs import create_training_data

#import data
(x_train, t_train), (x_test, t_test) = create_training_data()

#training
max_epochs = 10

network = SimpleConvNet(input_dim=(3,80,80), 
                        conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=2, weight_init_std=0.01)
                        
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=20,
                  optimizer='Adam', optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=100)

trainer.train()