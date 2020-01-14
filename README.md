# Genetic algorithm implementation
Parameters of the algorithm  
Population size  
Chromosome size  
Probability crossover  
Probability mutation  

Population consists of binary chromosome.  
Single point crossover.  
Roulette wheel type of selection for next chromosome for crossover.  
Algorithm terminates after maximum number of generations.  
A new generation has been produced where the population size has been reached and  
two worst chromosome from the new population has been replaced by the two best chromosomes from   
the old population.  

Testing  
The algorithm has been used for solving an instance of the Knapsack problem.
The file items.csv constist of table with columnd item,price,value.
The test run the GA and try to maximize the value if the price of the chosen elements
cannot be greater tha 5000.
Input Parameters  
chromosome size = 24 (number items in item.csv)  
population size = 100  
max generations 100
prob crossover 90%
prob mutation  0.01%

After the test pass a plot has been created.
In plot.PNG we see the best price,value for every new population.  
Since probability mutation of the algorithm is low it stuck in a global minimums for a while  
and suffer from low diversity.







