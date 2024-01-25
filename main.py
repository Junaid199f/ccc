import random
from numpy import asarray
import numpy as np
import random
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchsummary import summary
from de import DE

if __name__ == '__main__':
  # define lower and upper bounds for every dimension
  bounds = asarray(
    [(0, 0.99),(0, 3), (0, 0.99), (0, 3), (0, 0.99), (0, 3), (0, 0.99), (0, 3), (0, 0.99), (0, 3), (0, 0.99), (0, 3), (0, 0.99), (0, 3), (0, 0.99),
     (0, 3), (0, 0.99), (0, 3), (0, 0.99), (0, 3), (0, 0.99), (0, 3), (0, 0.99), (0, 3), (0, 0.99), (0, 3), (0, 0.99), (0, 3), (0, 0.99),
     (0, 3), (0, 0.99), (0, 3)])
  # define number of iterations
  iter = 100
  # define scale factor for mutation
  F = 0.5
  # define crossover rate for recombination
  cr = 0.7

  de = DE(5,2,0.9,0.6,32,10,3,1,32,20,16,0.3,bounds, iter, F, cr,False,False)
  de.evolve()