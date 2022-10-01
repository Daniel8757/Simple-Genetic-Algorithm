import numpy
import random
import math

import yaml

with open('config.yaml') as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)

DEFAULT_MUTATION = configs['default_mutation']
DEFAULT_CROSSOVER = configs['default_crossover']
DEFAULT_SIZE = configs['default_size']
MAX_GENERATIONS = configs['max_generations']

GAUSSIAN_VARIANCE = configs['gaussian_variance']

def make_population_zeroweights(size, length):
    zero_weight_num = math.floor(size*0.05)
    random_weight_num = size-zero_weight_num
    population = []

    for i in range(random_weight_num):
        new_genome = [random.uniform(-1, 1) for _ in range(length)]
        population.append(new_genome)

    for i in range(zero_weight_num):
        new_genome = [0 for _ in range(length)]
        population.append(new_genome)

    return population


class RealValueGA():
    def __init__(self, fitness, data_length, crossover_rate=DEFAULT_CROSSOVER, mutation_rate=DEFAULT_MUTATION, log=True):
        # fitness function
        self.fitness = fitness 
        # population size
        self.data_length = data_length
        # chance to crossover instead of moving directly onto next gen
        self.crossover_rate = crossover_rate
        # chance of a single gene mutation
        self.mutation_rate = mutation_rate
        # choice to print things to console
        self.log = log

    def log_print(self, to_output):
        if self.log:
            print(to_output)
    
    # average fitness of population
    def evaluate_fitness(self, population):
        total_fitness = 0
        for pop in population:
            total_fitness += self.fitness(pop)
        return total_fitness/len(population)

    # crossover one bit with equal probability
    def uniform_crossover(self, genome1, genome2):
        new_genome_1 = []
        new_genome_2 = []
        for i in range(len(genome1)):
            if random.random() < 0.5:
                new_genome_1.append(genome1[i])
                new_genome_2.append(genome2[i])
            else:
                new_genome_1.append(genome2[i])
                new_genome_2.append(genome1[i])
        return new_genome_1, new_genome_2

    # mutate a genome
    def mutate(self, genome):
        for i in range(len(genome)):
            if random.random() < self.mutation_rate:
                change = numpy.random.normal(scale=math.sqrt(GAUSSIAN_VARIANCE))
                genome[i] += change
                if genome[i] > 1:
                    genome[i] = 1
                if genome[i] < -1:
                    genome[i] = -1
        return genome
    
    # select two genomes using the fitness-proportionate selection
    def select_pair(self, population, weights):
        # edge case where the fitness of all pops are 0
        indexes = [i for i in range(len(population))]
        if not weights:
            choice = numpy.random.choice(indexes), numpy.random.choice(indexes)
        else:
            choice = numpy.random.choice(indexes, p=weights), numpy.random.choice(indexes, p=weights)

        return population[choice[0]], population[choice[1]]

    # get a weighting array for 
    def get_weight_arr(self, population):
        total_weight = 0
        for pop in population:
            total_weight += self.fitness(pop)

        if total_weight == 0:
            return None

        weights = list()
        for pop in population:
            weights.append(self.fitness(pop)/total_weight)

        return weights
    
    def fitness_proportion(self, curr_population):
        weights = self.get_weight_arr(curr_population)
        new_population = list()
        while len(new_population) < len(curr_population):
            p1, p2 = self.select_pair(curr_population, weights)
            if random.random() < self.crossover_rate:
                new_p1, new_p2 = self.uniform_crossover(p1, p2)
            else:
                new_p1, new_p2 = p1, p2
                
            new_p1 = self.mutate(new_p1)
            new_p2 = self.mutate(new_p2)

            new_population.append(new_p1)
            new_population.append(new_p2)
        return new_population

    # returns number of runs to get best genome
    def run(self, population, max_generations=MAX_GENERATIONS):
        self.log_print("Population size: {}".format(len(population)))
        self.log_print("Genome length: {}".format(self.data_length))
        curr_population = population
        for generation in range(max_generations):
            self.log_print("Generation {}: average fitness {}, best fitness {}".format(
                generation, 
                self.evaluate_fitness(curr_population), max(curr_population, key=self.fitness))
            )
            curr_population = self.fitness_proportion(curr_population)
            #TODO: create a stopping condition
        return None

class LinearRegression:
    # xvals should be a list of lists
    # yvals should be just a list
    def normalize(self, xvals, yvals):
        x_lambdas = []
        # transformation for each xvals 
        for i in range(len(xvals[0])):
            xvals_one = [weight[i] for weight in xvals]
            x_transform = lambda x : (x - min(xvals_one))/(max(xvals_one)-min(xvals_one))
            x_lambdas.append(x_transform)
        
        x_transformations = []
        for xval in xvals:
            x_transformations.append([x_lambdas[j](xval[j]) for j in range(len(xval))])
        
        y_transform = lambda y : (y - min(yvals))/(max(yvals)-min(yvals))
        y_transformation = [y_transform(one_y) for one_y in yvals]
        return x_transformations, y_transformation

    def loss(self, pred, actual):
        n = len(pred)
        squared_dist = sum([(pred[i]-actual[i])**2 for i in range(len(pred))])
        return squared_dist/n

    def run(self, dimensions, xvals, yvals, size=DEFAULT_SIZE, population=None, max_generations=MAX_GENERATIONS, crossover_rate=DEFAULT_CROSSOVER, mutation_rate=DEFAULT_MUTATION, log=True):
        if not population:
            population = make_population_zeroweights(size, dimensions+1)
        norm_xs, norm_ys = self.normalize(xvals, yvals)

        def fitness(genome):
            pred = []
            for xval in norm_xs:
                prediction = sum([xval[i]*genome[i] for i in range(len(xval))]) + genome[-1]
                pred.append(prediction)
            actual = norm_ys
            loss_value = self.loss(pred, actual)

            return 1/loss_value
        
        ga = RealValueGA(fitness, dimensions+1, crossover_rate, mutation_rate, log)
        ga.run(population, max_generations)

