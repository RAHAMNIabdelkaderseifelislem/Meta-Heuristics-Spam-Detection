"""
this file is for the shark smell optimization algorithm
"""

import random
import pandas as pd
from deap import base, creator, tools, algorithms
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer


class SharkSmellOptimization:
    def __init__(self, crossover_prob, mutation_prob, dimension, data_file):
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.dimension = dimension
        self.data_file = data_file
        self.data = self.load_data()
        self.toolbox = self._create_toolbox()

    def load_data(self):
        return pd.read_csv(self.data_file)

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
        print(self.data)
        subset_data = self.data['text']
        # Assume the last column is the target
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(subset_data)

        # Split the data into training and testing sets
        y = subset_data['spam']
        
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train a Naive Bayes classifier
        clf = MultinomialNB()
        clf.fit(X_train, y_train)
        
        # Predict on the test set and calculate accuracy
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Return accuracy as fitness
        return accuracy,

    def optimize(self, num_generations, population_size):
        pop = self.toolbox.population(n=population_size)
        hof = tools.HallOfFame(1)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        pop, logbook = algorithms.eaSimple(pop, self.toolbox, cxpb=self.crossover_prob, mutpb=self.mutation_prob, ngen=num_generations, stats=stats, halloffame=hof, verbose=True)

        return pop, logbook, hof
    
if __name__ == '__main__':
    sso = SharkSmellOptimization(crossover_prob=0.5, mutation_prob=0.2, dimension=10, data_file='data/emails.csv')
    pop, log, hof = sso.optimize(num_generations=10, population_size=50)
    print(hof)
    print(hof[0])
    print(hof[0].fitness.values[0])