"""
this file is for the wolf pack optimization algorithm
psudo code
1. Initialize the wolf pack with random positions in the search space.
2. For each wolf, calculate its fitness using the objective function.
3. Sort the wolves based on their fitness.
4. For each iteration:
    1. Select the alpha wolf (the one with the highest fitness).
    2. For each wolf in the pack:
        1. Calculate the distance to the alpha wolf.
        2. If the distance is less than a certain threshold, move the wolf towards the alpha wolf.
        3. If the distance is greater than the threshold, move the wolf in a random direction.
    3. For each wolf, calculate its new fitness.
    4. If a wolf's fitness is higher than the alpha wolf's fitness, it becomes the new alpha wolf.
5. Repeat the iterations until a stopping condition is met (e.g., maximum number of iterations, or a satisfactory fitness level is reached).
6. The position of the alpha wolf is the solution to the optimization problem.
"""
import random
import pandas as pd
from deap import base, creator, tools, algorithms
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer


class WolfPackOptimization:
    def __init__(self, population_size, generations, crossover_prob, mutation_prob, dimension, data_file):
        self.population_size = population_size
        self.generations = generations
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
        # Extract selected features from the dataset
        selected_features = [col for col, val in zip(self.data.columns, individual) if val]
        # Subset the data with selected features
        subset_data = self.data

        # Placeholder for model training and evaluation
        accuracy = self.train_and_evaluate_model(subset_data, self.data['spam'])
        return accuracy

    def train_and_evaluate_model(self, subset_data, labels):
        print(subset_data)
        # Assuming 'text' is the column containing text data
        text_data = subset_data['text']

        # Convert text data to numerical representation using TF-IDF
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(text_data)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

        # Replace this with your actual machine learning model and training logic
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        
        # print the accuracy of the model
        print("Accuracy of the model is: ", model.score(X_test, y_test))

        # Make predictions on the test set
        predictions = model.predict(X_test)

        # Evaluate the model (you may want to use other metrics depending on your problem)
        accuracy = accuracy_score(y_test, predictions)
        return accuracy

    def run(self):
        population = self.toolbox.population(n=self.population_size)

        algorithms.eaMuPlusLambda(population, self.toolbox, mu=self.population_size, lambda_=self.population_size,
                                  cxpb=self.crossover_prob, mutpb=self.mutation_prob, ngen=self.generations,
                                  stats=None, halloffame=None, verbose=True)

        best_individual = tools.selBest(population, k=1)[0]
        best_features = [i for i, val in enumerate(best_individual) if val]

        return best_features

# Example usage
if __name__ == "__main__":
    # Set your CSV file path
    csv_file_path = "data/emails.csv"

    # Set your dataset dimension (number of features)
    dataset_dimension = 100

    wpo = WolfPackOptimization(
        population_size=50,
        generations=100,
        crossover_prob=0.8,
        mutation_prob=0.2,
        dimension=dataset_dimension,
        data_file=csv_file_path
    )

    selected_features = wpo.run()
    print("Selected features:", selected_features)
