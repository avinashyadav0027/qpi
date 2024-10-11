import random
from distillaton import evaluate_fitness, generate_distill_data
from deap import base, creator, tools


# User input for GPU RAM and expected latency
gpu_ram = int(input("Enter your GPU RAM available: "))
expected_latency = float(input("Enter your expected latency: "))

# Define fitness and individual structure
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximize fitness
creator.create("Individual", list, fitness=creator.FitnessMax)

generate_distill_data()

# Toolbox for genetic operations
toolbox = base.Toolbox()

# Value sets for individual creation
first_value_set = list(range(96, 601, 12))  
second_value_set = [2, 4, 6, 8]

# Individual creation
def create_individual():
    first_value = random.choice(first_value_set)
    second_value = random.choice(second_value_set)
    return creator.Individual([first_value, second_value])

toolbox.register("individual", create_individual)  # Individual generation
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Evaluation function
 

toolbox.register("evaluate", evaluate_fitness, gpu_ram=gpu_ram, expected_latency=expected_latency)

# Crossover function
def custom_mate(ind1, ind2):
    if random.random() < 0.5:  
        ind1[0], ind2[0] = ind2[0], ind1[0]  
    else:  
        ind1[1], ind2[1] = ind2[1], ind1[1] 

# Mutation function
def custom_mutate(individual):
    individual[0] = random.choice(first_value_set)
    individual[1] = random.choice(second_value_set)

toolbox.register("mate", custom_mate)
toolbox.register("mutate", custom_mutate)
toolbox.register("select", tools.selTournament, tournsize=3)

# Initialize population
population = toolbox.population(n=10)  

# Number of generations to run the evolutionary algorithm
number_of_generations = 5

# Evolutionary algorithm loop
for gen in range(number_of_generations):
    print(f"-- Generation {gen} --")
    
    # Evaluate individuals in the population
    for individual in population:
        individual.fitness.values = toolbox.evaluate(individual)

    # Select the next generation individuals
    offspring = toolbox.select(population, len(population))  
    offspring = list(map(toolbox.clone, offspring)) 

    # Perform crossover and mutation on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < 0.5:  
            toolbox.mate(child1, child2)
            del child1.fitness.values  
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < 0.2:  
            toolbox.mutate(mutant)
            del mutant.fitness.values  

    # Evaluate the offspring
    for individual in offspring:
        if not individual.fitness.valid:
            individual.fitness.values = toolbox.evaluate(individual)

    # Replace population with the new generation
    population[:] = offspring

# Print final population and their fitness
for ind in population:
    print(ind, ind.fitness.values)
