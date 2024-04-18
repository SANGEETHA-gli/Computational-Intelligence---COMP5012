!pip install deap
!pip install matplotlib
!pip install numpy
!pip install pandas

import random
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt


POP_SIZE = 100
MAX_GEN = 100
CXPB = 0.8
MUTPB = 0.1
BIN_CAPACITY = 100  
NUM_ITEMS_PER_BIN = 3  

df = pd.read_csv('/content/sample_data/data8.txt', header=None, names=['Item_Size'])
items = df['Item_Size'].astype(int).tolist()


creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))  
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()


toolbox.register("attr_item", random.choice, items)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_item, n=len(items))


toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evaluate(individual):
    bins = [[] for _ in range(len(individual) // 3)]
    bin_loads = [0] * len(bins)

    for i, item_size in enumerate(individual):
        bin_index = i // 3
        if bin_loads[bin_index] + item_size <= BIN_CAPACITY:
            bins[bin_index].append(item_size)
            bin_loads[bin_index] += item_size

   
    num_bins_used = len(bins)

   
    max_load = max(bin_loads)
    avg_load = sum(bin_loads) / len(bins) if len(bins) > 0 else 0
    load_balance = -(max_load - avg_load)


    num_items_penalty = sum(1 for b in bins if len(b) != NUM_ITEMS_PER_BIN)

    return num_bins_used + num_items_penalty, load_balance

toolbox.register("evaluate", evaluate)


toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selNSGA2)


def main():
    pop = toolbox.population(n=POP_SIZE)

 
    archive = []

   
    for gen in range(MAX_GEN):
       
        offspring = toolbox.select(pop, len(pop))

       
        offspring = list(map(toolbox.clone, offspring))

        
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

       
        archive.extend([ind.fitness.values for ind in offspring if ind.fitness.valid and ind.fitness.values not in archive])

       
        pop[:] = offspring

   
    print("Solution Representations:")
    for ind in pop:
        print(ind)

   
    fits = [ind.fitness.values for ind in pop]
    num_bins, load_balance = zip(*fits)
    num_bins = np.array(num_bins)
    load_balance = np.array(load_balance)

   
    plt.scatter(num_bins, load_balance)
    plt.xlabel('Number of Bins Used')
    plt.ylabel('Load Balance')
    plt.title('Pareto Front')
    plt.show()

if __name__ == "__main__":
    main()
