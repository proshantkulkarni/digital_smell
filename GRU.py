import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange, repeat


class Model1(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, num_classes):
    super().__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.num_classes = num_classes
    self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_size * num_layers, self.num_classes)

  def forward(self, x):
    output , h = self.gru(x)
    h = rearrange(h, 'd n c -> n (d c)')
    return self.fc(h)

m = Model1(11, 32, 1, 17).cuda()
X = X.cuda()
m(X).shape