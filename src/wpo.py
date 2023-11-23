"""
this file is for the wolf pack optimization algorithm
"""
import random
from deap import base, creator, tools

class WolfPackOptimization:
    def __init__(self, population_size, generations, crossover_prob, mutation_prob, dimension):
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.dimension = dimension
        self.toolbox = self._create_toolbox()

    def _create_toolbox(self):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("attr_bool", random.randint, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=self.dimension)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", self.evaluate)

        return toolbox

    def evaluate(self, individual):
        # This is where you should put your spam detection model and evaluation logic
        # You need to train and evaluate a classifier using the selected features
        # The returned value should be the performance metric you want to maximize (e.g., accuracy)

        # For demonstration purposes, let's use a dummy fitness function
        fitness = sum(individual)
        return fitness,

    def run(self):
        population = self.toolbox.population(n=self.population_size)

        algorithms.eaMuPlusLambda(population, self.toolbox, mu=self.population_size, lambda_=self.population_size,
                                  cxpb=self.crossover_prob, mutpb=self.mutation_prob, ngen=self.generations,
                                  stats=None, halloffame=None, verbose=True)

        best_individual = tools.selBest(population, k=1)[0]
        best_features = [i for i, val in enumerate(best_individual) if val]

        return best_features
