from ga import GA
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import time
from itertools import count

def visualize(arr):
    plt.plot(arr)
    plt.ylabel('fitness score')
    plt.xlabel("nth generation")
    plt.show()

def main():
    chromosome_size = 24
    population_size = 100
    constraint = 5000
    prob_crossover = 0.9
    prob_mutation = 0.001
    max_generations = 100
    solver = GA(chromosome_size,population_size,constraint,prob_crossover,prob_mutation,max_generations)
    Results = []
    chromosome  = 0
    i = count()
    while(solver.termination_test()):
        solver.generate_new_population()
        chromosome, score = solver.get_best_fitness_chromosome()
        Results.append(score)

    items, total_price, total_value = solver.convert_chromosome_to_item_list(chromosome)
    print(items)
    print()
    print(f"Total price : {total_price}")
    print(f"Total value: {total_value}" )
    visualize(Results)


def main_test():
    chromosome_size = 24
    population_size = 4
    constraint = 5000
    prop_crossover = 0.9
    prob_mutation = 0.01
    max_generations = 3
    solver = GA(chromosome_size,population_size,constraint,prop_crossover,prob_mutation,max_generations)
    chromosome = np.random.choice([0,1],(chromosome_size,population_size))

    print(solver.calculate_fitness())
    #print(solver.population)

if __name__ == "__main__":
    main()
