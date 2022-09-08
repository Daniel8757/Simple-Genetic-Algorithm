import random
import numpy

class SimpleGA():
    def __init__(self, fitness, data_length=20, crossover_rate=0.07, mutation_rate=0.001):
        # fitness function
        self.fitness = fitness 
        # population size
        self.data_length = data_length
        # chance to crossover instead of moving directly onto next gen
        self.crossover_rate = crossover_rate
        # chance of a single gene mutation
        self.mutation_rate = mutation_rate
    
    # average fitness of population
    def evaluate_fitness(self, population):
        total_fitness = 0
        for pop in population:
            total_fitness += self.fitness(pop)
        return total_fitness/len(population)

    def crossover(self, genome1, genome2):
        # crossover point is uniformly distributed over the length
        cross_point = random.randint(0, self.data_length)
        new_genome = genome1[:cross_point] + genome2[cross_point:]
        return new_genome

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
    
    # finds the best genome after num_runs generations
    def run(self, population, num_runs=50):
        print("Population size: {}".format(len(population)))
        print("Genome length: {}".format(self.data_length))
        curr_population = population
        for generation in range(num_runs):
            print("Generation {}: average fitness {}, best fitness {}".format(
                generation, 
                self.evaluate_fitness(curr_population), max(curr_population, key=self.fitness))
            )
            weights = self.get_weight_arr(curr_population)
            new_population = list()
            while len(new_population) < len(population):
                p1, p2 = self.select_pair(curr_population, weights)
                if random.random() < self.crossover_rate:
                    new_p1 = self.crossover(p1, p2)
                    new_p2 = self.crossover(p1, p2)
                else:
                    new_p1 = p1
                    new_p2 = p2
                new_p1 = self.mutate(new_p1)
                new_p2 = self.mutate(new_p2)

                new_population.append(new_p1)
                new_population.append(new_p2)
            curr_population = new_population
        return max(curr_population, key=self.fitness)

class MaximizeOnes():
    # create a random genome
    def random_genome(self, length):
        ret = ""
        for _ in range(length):
            if random.randint(0, 1) == 0:
                ret += "0"
            else:
                ret += "1"
        return ret

    # create a population of genomes
    def make_population(self, size, length):
        population = list()
        for _ in range(size):
            population.append(self.random_genome(length))
        return population

    # utility function
    def fitness(self, genome):
        utility = 0
        for c in genome:
            if c == '1':
                utility += 1
        return utility

    def run(self, generations=200, population=None, size=50, length=20, crossover_rate=0.07, mutation_rate=0.001):
        if not population:
            population = self.make_population(size, length)
        
        ga = SimpleGA(self.fitness, length, crossover_rate, mutation_rate)
        result = ga.run(population, generations)
        return result

def main():
    optimize = MaximizeOnes()
    optimum = optimize.run()
    print("the best candidate is: {}".format(optimum))

if __name__ == "__main__":
    main()