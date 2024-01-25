import augment
import augmentations
from dataset import Dataset
from evaluate import Evaluate
import genotype
import operations
from operations_mapping import operations_mapping
from population import Population
import utils
import pickle
from Surrogate import Surrogate

class Optimizer:
  def __init__(self, population_size, number_of_generations, crossover_prob, mutation_prob, blocks_size, num_classes,
               in_channels, epochs, batch_size, layers, n_channels, dropout_rate,bounds, iter, F, cr, retrain,resume_train):
    self.resume_train = resume_train
    if self.resume_train==True:
      z = open('checkpoints/checkpoints.pkl','rb')
      #self.pop =  Population(blocks_size, population_size,bounds)
      self.pop = pickle.load(z)
    else:
      self.pop = Population(blocks_size, population_size,bounds)
    self.population_size = population_size
    self.iter = iter
    self.bounds = bounds
    self.F = F
    self.cr = cr
    self.surrogate = Surrogate()
    self.number_of_generations = number_of_generations
    self.crossover_prob = crossover_prob
    self.epochs = epochs
    self.mutation_prob = mutation_prob
    self.batch_size = batch_size
    self.blocks_size = blocks_size
    self.num_classes = num_classes
    self.in_channels = in_channels
    self.layers = layers
    self.dropout_rate = dropout_rate
    self.n_channels = n_channels
    self.intermediate_channels = [2 ** (i + 1) for i in range(1, blocks_size + 1, 1)]
    self.retrain = retrain
    self.fitness = []
    self.networks = []
    self.trained_individuals = False
    self.decoded_individuals = []
    self.offsprings_population = []
    self.offsprings_fitness = []
    self.evaluator = Evaluate(self.batch_size)

    def evolve():
      pass

  def encode(self):
    return None

  def decode(self):
    self.decoded_individuals = self.pop.decode_individuals(self.pop.Individuals)
    # for i in range(self.population_size):
    # model = Net(pop.individual[0],self.in_channels,self.intermediate_channels,self.num_classes,self.block_size)

  def optimize(self):
    self.evolve()



 