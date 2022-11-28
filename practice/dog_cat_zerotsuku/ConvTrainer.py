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
from LoadDogs import *

# Load dogs and cats
training_data = create_training_data()


max_epochs = 5

network = SimpleConvNet(input_dim=(1,28,28,3), 
                        conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=2, weight_init_std=0.01)
                        
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=100)
trainer.train()

# パラメータの保存
network.save_params("params.pkl")
print("Saved Network Parameters!")

# グラフの描画
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
