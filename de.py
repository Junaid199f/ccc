import json
import random
import os
import numpy as np
import random
import torch
import torchvision
import hashlib
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchsummary import summary
from numpy import savetxt
from datetime import datetime
import pandas as pd
import random
import pickle
from numpy.random import rand
from numpy.random import choice
from numpy import asarray
from numpy import clip
from numpy import argmin
from numpy import min
from numpy import around
from matplotlib import pyplot
import random
from model import NetworkCIFAR
from utils import decode_cell, decode_operations
from optimizer import Optimizer


class DE(Optimizer):
  def __init__(self,population_size,number_of_generations,crossover_prob,mutation_prob,blocks_size,num_classes,in_channels,epochs,batch_size,layers,n_channels,dropout_rate,bounds, iter, F, cr, retrain,resume_train):
    super().__init__(population_size,number_of_generations,crossover_prob,mutation_prob,blocks_size,num_classes,in_channels,epochs,batch_size,layers,n_channels,dropout_rate,bounds, iter, F, cr,retrain,resume_train)
  #Convert ParamsChoice values to int of individuals
  def convert_encoding_to_int(self,indv):
    indv_int = []
    for i in range(len(indv)):
      if i%2!=0:
        indv_int.append(int(indv[i]))
      else:
        indv[i] = round(indv[i],2)
        indv_int.append(indv[i])
    return indv_int
  # define objective function
  def obj(self,x):
    x= self.convert_encoding_to_int(x)
    print(x)
    decoded_indv = NetworkCIFAR(self.n_channels, self.num_classes, self.layers, True,
                                 decode_cell(decode_operations(x, self.pop.indexes)), self.dropout_rate, 'FP32',
                                 False)
    hash_indv = hashlib.md5(str(x).encode("UTF-8")).hexdigest()
    loss = self.evaluator.train(decoded_indv, self.epochs, hash_indv)
    #print(x)
    return loss

  # define mutation operation
  def mutation(self,x, F):
    #print(x)
    return x[0] + F * (x[1] - x[2])

  # define boundary check operation
  def check_bounds(self,mutated, bounds):
    mutated_bound = [clip(mutated[i], bounds[i, 0], bounds[i, 1]) for i in range(len(bounds))]
    return mutated_bound

  # define crossover operation
  def crossover(self,mutated, target, dims, cr):
    # generate a uniform random value for every dimension
    p = rand(dims)
    # generate trial vector by binomial crossover
    trial = [mutated[i] if p[i] < cr else target[i] for i in range(dims)]
    return trial

  def evolve(self):

    obj_all = [self.obj(ind) for ind in self.pop.individuals]
    print(obj_all)
    print(self.pop.individuals)
    train_data = np.asarray(self.pop.individuals)
    label = np.asarray(obj_all)

    self.surrogate.gbm_regressor(train_data,label)
    # find the best performing vector of initial population
    best_vector = self.pop.individuals[argmin(obj_all)]
    best_obj = min(obj_all)
    prev_obj = best_obj
    # initialise list to store the objective function value at each iteration
    obj_iter = list()
    # run iterations of the algorithm
    for i in range(self.number_of_generations):
      # iterate over all candidate solutions
      for j in range(self.population_size):
        # choose three candidates, a, b and c, that are not the current one
        candidates = [candidate for candidate in range(self.population_size) if candidate != j]
        a, b, c = self.pop.individuals[choice(candidates, 3, replace=False)]
        # perform mutation
        mutated = self.mutation([a, b, c], self.F)
        # check that lower and upper bounds are retained after mutation
        mutated = self.check_bounds(mutated, self.bounds)
        # perform crossover
        trial = self.crossover(mutated, self.pop.individuals[j], len(self.bounds), self.cr)
        # compute objective function value for target vector
        obj_target = self.obj(self.pop.individuals[j])
        # compute objective function value for trial vector
        obj_trial = self.obj(trial)
        # perform selection
        if obj_trial < obj_target:
          # replace the target vector with the trial vector
          self.pop.individuals[j] = trial
          # store the new objective function value
          obj_all[j] = obj_trial
      # find the best performing vector at each iteration
      best_obj = min(obj_all)
      # store the lowest objective function value
      if best_obj < prev_obj:
        best_vector = self.pop.individuals[argmin(obj_all)]
        prev_obj = best_obj
        obj_iter.append(best_obj)
        # report progress at each iteration
        print('Iteration: %d f([%s]) = %.5f' % (i, around(best_vector, decimals=5), best_obj))
    return [best_vector, best_obj, obj_iter]
