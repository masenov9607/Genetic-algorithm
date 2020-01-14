import numpy as np
import pandas as pd

class GA(object):
    def __init__(self,chromosome_size,population_size,constraint,prop_crossover,prob_mutation,max_generations):
        self.pc = prop_crossover
        self.pm = prob_mutation
        self.max_generations = max_generations
        self.population_size = population_size
        self.chromosome_size = chromosome_size
        self.population = np.random.choice([0,1],(population_size, chromosome_size))
        self.constraint = constraint
        self.items = pd.read_csv("items.csv", index_col="ID", sep=";")

    def fitness(self,chromosome):
        price = self.items[chromosome == 1]["price"].sum()
        while price > self.constraint:
            ones_indices            = np.nonzero(chromosome)[0].tolist()
            index                   = np.random.choice(ones_indices, 1, replace=False)
            chromosome[index]       = 0
            price                   = self.items[chromosome == 1]["price"].sum()

        fitness_score =  self.items[chromosome == 1]["value"].sum()
        return fitness_score

    def calculate_fitness(self):
        result = 0
        for x in np.ndindex(self.population.shape[0]):
            result += self.fitness(self.population[x])
        return result

    def select(self):
         S = self.calculate_fitness()
         num = np.random.uniform(0,S,1)[0]
         P = 0
         for x in np.ndindex(self.population.shape[0]):
             chromosome = self.population[x]
             P += self.fitness(chromosome)
             if P > num:
                 return chromosome

    def crossover(self, firts_parent,second_parent):
        first_child = firts_parent
        second_child = second_parent
        if np.random.uniform(0,1,1)[0] < self.pm:
            i = np.random.choice(firts_parent.shape[0], 1, replace=False)[0]
            first_child = np.concatenate([firts_parent[:i],second_parent[i:]])
            second_child = np.concatenate([firts_parent[:i],second_parent[i:]])
        return (first_child,second_child)

    def mutate(self,chromosome):
        for i in range(self.chromosome_size):
            if np.random.uniform(0,1,1)[0] < self.pm:
                chromosome[i] = (chromosome[i] + 1) % 2

    def termination_test(self):

        return self.max_generations != 0

    def get_best_fitness_chromosome(self):
        max_score = 0
        for x in range(self.population_size):
            chromosome = self.population[x]
            fitness_score = self.fitness(chromosome)
            if max_score < fitness_score:
                max_score = fitness_score
                best_fitness_chromosome = chromosome
        return (best_fitness_chromosome,max_score)

    def get_worst_fitness_chromosome(self):
        min_score = self.fitness(self.population[0])
        worst_fitness_chromosome = None
        i = 0
        for x in range(self.population_size):
            chromosome = self.population[x]
            fitness_score = self.fitness(chromosome)
            if min_score > fitness_score:
                min_score = fitness_score
                worst_fitness_chromosome = chromosome
                i = x
        return (i,worst_fitness_chromosome,min_score)

    def generate_new_population(self):
        self.max_generations = self.max_generations - 1
        next_population = []
        i = 0

        while(len(next_population) < self.population_size):
            parent1 = self.select()
            parent2 = self.select()
            while np.array_equal(parent1,parent2):
                parent2 = self.select()
            child1,child2 = self.crossover(parent1,parent2)
            self.mutate(child1)
            self.mutate(child2)
            next_population.append(child1)
            next_population.append(child2)
            #print("New child added to the population " + str(i))
            i += 2
        best_fitness_chromosome = self.get_best_fitness_chromosome()[0]
        self.population = np.array(next_population)

        for x in range(self.population_size):
            chromosome = self.population[x]
            if np.array_equal(best_fitness_chromosome,chromosome):
                break
        else:
            wort_chromosome_index = self.get_worst_fitness_chromosome()[0]
            self.population[wort_chromosome_index] = best_fitness_chromosome

    def convert_chromosome_to_item_list(self,chromosome):
        chosen_items = self.items.loc[chromosome == 1]
        total_price = chosen_items["price"].sum()
        total_value = chosen_items["value"].sum()
        return chosen_items,total_price,total_value
