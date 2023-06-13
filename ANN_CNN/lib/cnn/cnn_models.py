from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.mlp.fully_conn import *
from lib.mlp.layer_utils import *
from lib.cnn.layer_utils import *


""" Classes """
class TestCNN(Module):
    def __init__(self, dtype=np.float32, seed=None):
        self.net = sequential(
            ########## TODO: ##########
            ConvLayer2D(3, 3, 3, 1, 0, name='conv1'),
            MaxPoolingLayer(2,2, name='pool1'),
            flatten(name='flatten'),
            fc(27, 5, 0.02, name='fc1'),
            ########### END ###########
        )


class SmallConvolutionalNetwork(Module):
    def __init__(self, keep_prob=0, dtype=np.float32, seed=None):
        self.net = sequential(
            ########## TODO: ##########
            ConvLayer2D(3, 3, 16, 1, 0, name = "conv1"),
            gelu(name="lr1"),
            MaxPoolingLayer(3, 2, name = "pool1"),
            ConvLayer2D(16, 3, 16, 1, 0, name = "conv2"),
            gelu(name="lr2"),
            MaxPoolingLayer(3, 1, name = "pool2"),
            flatten(name = "flatten1"),
            fc(1600, 100, 0.02, name="fc1"),
            gelu(name="lr3"),
            dropout(keep_prob=0.5, seed=seed, name="dropout1"),
            fc(100, 20, 0.02, name="fc2"),
            ########### END ###########
        )