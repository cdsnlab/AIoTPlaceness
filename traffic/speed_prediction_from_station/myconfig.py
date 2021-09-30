import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable
import torchvision
from torchvision import transforms
from torchvision import datasets
import numpy as np
import copy

import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

from mymodels import *

models = {
    "OurModelFastM11": OurModelFastM11.OurModelFastM11,
    "OurModelFastM10": OurModelFastM10.OurModelFastM10,
    "OurModelFastM20": OurModelFastM20.OurModelFastM20,
    "OurModelFastM30": OurModelFastM30.OurModelFastM30,
    "OurModelFastM40": OurModelFastM40.OurModelFastM40,
    "OurModelFastM50": OurModelFastM50.OurModelFastM50,
    "OurModelFastM60": OurModelFastM60.OurModelFastM60,
    "OurModelFastM70": OurModelFastM70.OurModelFastM70,
    "OurModelFastM80": OurModelFastM80.OurModelFastM80,
    "OurModelCustom": OurModelCustom.OurModelCustom,
    "OurModelFastS": OurModelFastS.OurModelFastS,
    "OurModelFastM": OurModelFastM.OurModelFastM,
    "OurModelFast": OurModelFast.OurModelFast,
    "FCN": FCN.FCN,
}

testdir = 'torchmodel_20210419_N_G'
dataset = 'G'
num_epoch = 3000
epoch_decay = 200
learning_rate = 0.002
early_stop = False
early_iter = 300
gpu = 0
