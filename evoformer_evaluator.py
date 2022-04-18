import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import sys
from tqdm import tqdm
from einops import rearrange

from Evo_Dataset import Evo_Dataset
from Models import Evoformer

# CONSTANTS
batch_size = 4
r = 64
c_m = 128
c_z = 64
c = 8
s = 8

stride = 64
num_epochs = 100
learning_rate = 0.01
progress_bar = True
save_to_file = True
load_from_file = False
USE_DEBUG_DATA = True