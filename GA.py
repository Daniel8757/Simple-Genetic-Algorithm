import random
import numpy
import math
import heapq

import yaml

with open('config.yaml') as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)

DEFAULT_MUTATION = configs['default_mutation']
DEFAULT_CROSSOVER = configs['default_crossover']
DEFAULT_SIZE = configs['default_size']
MAX_RUNS = configs['max_runs']
ELITISM_NUMBER = configs['elitism_number']

CROSSOVER_TYPE = configs['crossover_type']
SELECTION_TYPE = configs['selection_type']

DEFAULT_LENGTH_MONES = configs['default_length_mones']
FITNESS_TYPE_MONES = configs['fitness_type_mones']

DEFAULT_LENGTH_RASTRIGIN = configs['default_length_rastrigin']
MAX_FITNESS = configs['max_fitness']

#TODO: add error checking

def random_genome(length):
    ret = ""
    for _ in range(length):
        if random.randint(0, 1) == 0:
            ret += "0"
        else:
            ret += "1"
    return ret

# create a population of genomes
def make_population(size, length):
    population = list()
    for _ in range(size):
        population.append(random_genome(length))
    return population

class SimpleGA():
    def __init__(self, fitness, best_genome, data_length=DEFAULT_LENGTH_MONES, crossover_rate=DEFAULT_CROSSOVER, mutation_rate=DEFAULT_MUTATION, log=True):
        # fitness function
        self.fitness = fitness 
        # genome stopping condition
        self.best_genome = best_genome
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

    # crossover at one point
    def one_point_crossover(self, genome1, genome2):
        # crossover point is uniformly distributed over the length
        cross_point_1 = random.randint(0, self.data_length)
        cross_point_2 = len(genome1)-cross_point_1
        new_genome_1 = genome1[:cross_point_1] + genome2[cross_point_1:]
        new_genome_2 = genome2[:cross_point_2] + genome1[cross_point_2:]
        return new_genome_1, new_genome_2

    # crossover one bit with equal probability
    def uniform_crossover(self, genome1, genome2):
        new_genome_1 = ""
        new_genome_2 = ""
        for i in range(len(genome1)):
            if random.random() < 0.5:
                new_genome_1 += genome1[i]
                new_genome_2 += genome2[i]
            else:
                new_genome_1 += genome2[i]
                new_genome_2 += genome1[i]
        return new_genome_1, new_genome_2

    # mutate a genome
    def mutate(self, genome):
        gene_list = list(genome)
        for i in range(len(gene_list)):
            if random.random() < self.mutation_rate:
                if gene_list[i] == '1':
                    gene_list[i] = '0'
                else:
                    gene_list[i] = '1'
        return ''.join(gene_list)
    
    # select two genomes using the fitness-proportionate selection
    def select_pair(self, population, weights):
        # edge case where the fitness of all pops are 0
        if not weights:
            return numpy.random.choice(population), numpy.random.choice(population)
        
        return numpy.random.choice(population, p=weights), numpy.random.choice(population, p=weights)

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
                if CROSSOVER_TYPE == 'uniform':
                    new_p1, new_p2 = self.uniform_crossover(p1, p2)
                elif CROSSOVER_TYPE == 'onepoint':
                    new_p1, new_p2 = self.one_point_crossover(p1, p2)
                else:
                    new_p1, new_p2 = p1, p2
            else:
                new_p1, new_p2 = p1, p2
                
            new_p1 = self.mutate(new_p1)
            new_p2 = self.mutate(new_p2)

            new_population.append(new_p1)
            new_population.append(new_p2)
        return new_population

    def elitism(self, curr_population):
        # TODO: onfigure elitism parameter
        # heapify the curr_population to get N best genomes
        heapify_pops = list()
        for pop in curr_population:
            heapify_pops.append((-self.fitness(pop), pop))
        heapq.heapify(heapify_pops)

        best_pops = list()
        for _ in range(ELITISM_NUMBER):
            _, pop = heapq.heappop(heapify_pops)
            best_pops.append(pop)

        non_selected = [pop for _, pop in heapify_pops]

        return self.fitness_proportion(non_selected) + best_pops

    # returns number of runs to get best genome
    def run(self, population, max_runs=MAX_RUNS):
        self.log_print("Population size: {}".format(len(population)))
        self.log_print("Genome length: {}".format(self.data_length))
        curr_population = population
        for generation in range(max_runs):
            self.log_print("Generation {}: average fitness {}, best fitness {}".format(
                generation, 
                self.evaluate_fitness(curr_population), max(curr_population, key=self.fitness))
            )
            if SELECTION_TYPE == "elitism":
                curr_population = self.elitism(curr_population)
            elif SELECTION_TYPE == "proportion":
                curr_population = self.fitness_proportion(curr_population)
            if max(curr_population, key=self.fitness) == self.best_genome:
                return generation
        return None

class MaximizeOnes():
    # utility function
    def fitness_uniform(self, genome):
        utility = 0
        for c in genome:
            if c == '1':
                utility += 1
        return utility

    # utility function
    def fitness_exponential(self, genome):
        utility = 0
        curr = 1
        for c in genome:
            if c == '1':
                utility += curr
            curr *= 2
        return utility

    def run(self, max_runs=MAX_RUNS, population=None, size=DEFAULT_SIZE, length=DEFAULT_LENGTH_MONES, crossover_rate=DEFAULT_CROSSOVER, mutation_rate=DEFAULT_MUTATION, log=True):
        if not population:
            population = make_population(size, length)

        if FITNESS_TYPE_MONES == "exponential":
            fit_func = self.fitness_exponential
        elif FITNESS_TYPE_MONES == "uniform":
            fit_func = self.fitness_uniform

        best_genome = '1'*DEFAULT_LENGTH_MONES
        
        ga = SimpleGA(fit_func, best_genome, length, crossover_rate, mutation_rate, log)
        result = ga.run(population, max_runs)
        return result

class MinimizeRastrigin():
    def fitness(self, genome):
        num = 0
        curr_addition = 5.12/(2**DEFAULT_LENGTH_RASTRIGIN-1)
        for i in genome[:DEFAULT_LENGTH_RASTRIGIN]:
            if i == '1':
                num += curr_addition
            curr_addition *= 2
        
        if genome[0] == '0':
            num *= -1

        rastrigin_apply = 10 + num**2 - 10*math.cos(2*math.pi*num)
        return MAX_FITNESS - rastrigin_apply

    def run(self, max_runs=MAX_RUNS, population=None, size=DEFAULT_SIZE, crossover_rate=DEFAULT_CROSSOVER, mutation_rate=DEFAULT_MUTATION, log=True):
        if not population:
            population = make_population(size, DEFAULT_LENGTH_RASTRIGIN)

        best_genome = '0'*DEFAULT_LENGTH_RASTRIGIN
        
        ga = SimpleGA(self.fitness, best_genome, DEFAULT_LENGTH_RASTRIGIN, crossover_rate, mutation_rate, log)
        result = ga.run(population, max_runs)
        return result

def main():
    ga_object = MinimizeRastrigin()
    num_runs = ga_object.run()
    print("It takes {} runs".format(num_runs))
    pass

if __name__ == "__main__":
    main()