import GA
import matplotlib.pyplot as plt
import math

import yaml

with open('config.yaml') as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)

REMOVE_IF_NOT_FOUND = configs['remove_if_not_found']
DATAPOINTS_FOR_AVERAGE = configs['datapoints_for_average']
NUM_DATAPOINTS = configs['num_datapoints']

MAX_RUNS = configs['max_runs']

def test_mutation(GA_object):
    mutation_rates = []
    num_runs = []
    curr_mutation = 0.0001
    mul_rate = 8000 ** (1/NUM_DATAPOINTS)
    for _ in range(NUM_DATAPOINTS):
        total = 0
        for _ in range(DATAPOINTS_FOR_AVERAGE):
            one_run = GA_object.run(mutation_rate=curr_mutation, log=False)
            if one_run:
                total += one_run
            else:
                total += MAX_RUNS

        runs = total/DATAPOINTS_FOR_AVERAGE
        num_runs.append(runs)

        mutation_rates.append(curr_mutation)
        curr_mutation *= mul_rate
        
        print("mutation: {} average runs: {}".format(curr_mutation, runs))

    plt.plot(mutation_rates, num_runs, marker="o")
    plt.title('Histogram of IQ')
    plt.xlabel('Mutation Rate')
    plt.ylabel('Number of Runs')
    plt.show()

def test_crossover(GA_object):
    crossover_rates = []
    num_runs = []
    curr_crossover = 0
    for _ in range(NUM_DATAPOINTS):
        total = 0
        for _ in range(DATAPOINTS_FOR_AVERAGE):
            one_run = GA_object.run(crossover_rate=curr_crossover, log=False)
            if one_run:
                total += one_run
            else:
                total += MAX_RUNS

        runs = total/DATAPOINTS_FOR_AVERAGE
        num_runs.append(runs)

        crossover_rates.append(curr_crossover)
        curr_crossover += (1/NUM_DATAPOINTS)
        
        print("crossover: {} average runs: {}".format(curr_crossover, runs))

    plt.plot(crossover_rates, num_runs, marker="o")
    plt.title('Histogram of IQ')
    plt.xlabel('Crossover Rate')
    plt.ylabel('Number of Runs')
    plt.show()

def test_population(GA_object):
    populations = []
    num_runs = []
    curr_population = 15
    for _ in range(NUM_DATAPOINTS):
        total = 0
        for _ in range(DATAPOINTS_FOR_AVERAGE):
            one_run = GA_object.run(size=curr_population, log=False)
            if one_run:
                total += one_run
            else:
                total += MAX_RUNS

        runs = total/DATAPOINTS_FOR_AVERAGE
        num_runs.append(runs)

        populations.append(curr_population)
        curr_population += 1
        
        print("population: {} average runs: {}".format(curr_population, runs))

    plt.plot(populations, num_runs, marker="o")
    plt.title('Histogram of IQ')
    plt.xlabel('Population')
    plt.ylabel('Number of Runs')
    plt.show()

def test_mutation_and_crossover(GA_object):
    num_mutations = math.floor(NUM_DATAPOINTS ** 0.5)
    num_crossovers = math.floor(NUM_DATAPOINTS ** 0.5)

    mutations_arr = []
    crossovers_arr = []
    num_runs = []

    curr_mutation = 0.0001
    mul_rate = 8000 ** (1/NUM_DATAPOINTS)

    for _ in range(num_mutations):
        curr_crossover = 0
        for _ in range(num_crossovers):
            total = 0
            for _ in range(DATAPOINTS_FOR_AVERAGE):
                one_run = GA_object.run(mutation_rate=curr_mutation, crossover_rate=curr_crossover, log=False)
                if one_run:
                    total += one_run
                else:
                    total += MAX_RUNS

            runs = total/DATAPOINTS_FOR_AVERAGE
            num_runs.append(runs)

            mutations_arr.append(curr_mutation)
            crossovers_arr.append(curr_crossover)
            print("mutation: {} crossover: {} average runs: {}".format(curr_mutation, curr_crossover, runs))

            curr_crossover += (1/num_crossovers)
        curr_mutation *= mul_rate

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(mutations_arr, crossovers_arr, num_runs)
    ax.set_xlabel('Mutation Rate')
    ax.set_ylabel('Crossover Rate')
    ax.set_zlabel('Number of runs')

    plt.show()

test_population(GA.MinimizeRastrigin())