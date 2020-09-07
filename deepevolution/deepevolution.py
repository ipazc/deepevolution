import numpy as np
import pandas as pd
import tensorflow as tf


class DeepEvolution:

    def __init__(self, model):
        """
        Instantiates a DeepEvolution object that keep track of a population of models and allows to evolve them.

        :param model:
            keras model to evolve.
        """
        self._model = model
        self._generation = []

    @staticmethod
    def _build_random_model_weights(model):
        """
        Generates a randomized weights version from the given model.
        It is the model with some normal distributed "noise" in the weights, with a std deviation of 0.03.

        :param model:
            keras model to derive.
        :return:
            new model weights.
        """
        model_weights = model.get_weights()

        new_weights = []
        for layer_weights in model_weights:
            new_weights.append(layer_weights + np.random.normal(0, 0.03, size=layer_weights.shape))

        return new_weights

    def _adjust_generation(self, new_population):
        """
        Adjusts the generation list with the given population count.

        :param new_population:
            number of individuals to be held in the generation list.
        """
        if len(self._generation) > new_population:
            self._generation = self._generation[:new_population]

        elif len(self._generation) < new_population:
            if len(self._generation) == 0:
                self._generation.append(self._model.get_weights())

            for remaining_individual in range(new_population - len(self._generation)):
                self._generation.append(self._build_random_model_weights(self._model))

    def _find_elite(self, fitness_func, X, Y, top_k):
        """
        Finds the `top_k` elite members in the generation, given a fitness function `fitness_func` that tells how well
        are performing the models.

        :param fitness_func:
            function parameterized as `function(model, X, Y)` that tells a score for this model. Note that the score can be any
            value, and it doesn't need to be the result of a backpropagate-capable function.
            The higher the score, the better the model.

        :returns:
            a Series of `top_k` individuals sorted by the score in descending order (first is best, last is worst).
        """
        scores = []

        for weights in self._generation:
            self._model.set_weights(weights)
            fitness_score = fitness_func(self._model, X, Y)
            scores.append(fitness_score)

        weights_sorted = pd.Series(self._generation, index=scores).sort_index(ascending=False)
        return weights_sorted.iloc[:top_k]

    @staticmethod
    def _single_crossover(genitor1_weights, genitor2_weights, mutation_rate, mutation_std):
        """
        Performs a crossover between two individuals. This crossover mixes the 50% of the weights between both individuals,
        and applies a mutation to the result. The mutation consists of a noise sampled from a normal distribution with the given
        `mutation_std` std. This mutation is applied to the `mutation_rate` percent of the result weights.

        :param genitor1_weights:
            the 'father' of the baby.

        :param genitor2_weights:
            the 'mother' of the baby.

        :param mutation_rate:
            the % of weights to be mutated.

        :param mutation_std:
            the std of the normal distribution from where the mutation is sampled.
        """
        baby_weights = []

        for layer_id in range(len(genitor1_weights)):
            genitor1_layer = genitor1_weights[layer_id]
            genitor2_layer = genitor2_weights[layer_id]

            # Let's build the contribution mask for each genitor
            genitor1_contribution_mask = np.random.rand(*genitor1_layer.shape) < 0.5
            genitor2_contribution_mask = np.invert(genitor1_contribution_mask)

            # Now the mutation mask
            mutation_values = np.random.normal(0, mutation_std, genitor1_layer.shape)
            mutation_mask = np.random.rand(*genitor1_layer.shape) < mutation_rate

            # Then we apply the masks to build the baby layer weights
            baby_layer = genitor1_layer * genitor1_contribution_mask + genitor2_layer * genitor2_contribution_mask
            baby_layer_mutated = baby_layer + mutation_values * mutation_mask
            baby_weights.append(baby_layer_mutated)

        return baby_weights

    @staticmethod
    def _crossover(sorted_weights_list, mutation_rate, mutation_std):
        """
        Perform a crossover among the individuals of a list.

        The first individual (the most successful) is crossed over the rest of the individuals.
        The second individual is crossed over the rest of the individuals after him.
        ...
        The last individual is only crossed to himself.

        This ensures that the best performing model spreads its weights among the generation more than the rest of the population.

        :param sorted_weights_list:
            list of weights sorted by score (best performing first).

        :param mutation_rate:
            the % of weights to be mutated.

        :param mutation_std:
            the std of the normal distribution from where the mutation is sampled.

        :return:
            the list of babies generated.

        """
        babies = []

        for genitor1_index, genitor1_weights in enumerate(sorted_weights_list):

            for genitor2_index, genitor2_weights in enumerate(sorted_weights_list[genitor1_index:]):
                baby = DeepEvolution._single_crossover(genitor1_weights, genitor2_weights, mutation_rate, mutation_std)
                babies.append(baby)

        return babies

    def evolve(self, X, Y, max_generations=40, fitness_func=None, population=16, top_k=4, mutation_rate=0.2,
               mutation_std=0.03):
        """
        Evolves the model for the specified data and fitness function.

        For the given model, it will generate a population of `population` weights (randomly derived from the model) and then
        evolve them until the generation `max_generations` is met. The evolution consists of crossing the best performing models
        and discarding the worst performing models.

        This method is a generator, so it can be iterated.

        :param X:
            The training data to be used for the fitness function. This data is only used by the `fitness_func` function and
            depends totally on what this function does.

        :param Y:
            The training data labels to be used for the fitness function. As the X parameter, this data is only used by
            the `fitness_func` function and depends totally on what this function does.

        :param max_generations:
            The number of generations to execute the evolution.

        :param fitness_func:
            The fitness function. It is a function with the prototype `function(model, X, Y)`. This function must do something
            with the model to score it. It might be, for example, evaluating the model with the data (X,Y) and telling an accuracy
            or any other value that tells how well it is performing. Note that it is not required to be a backpropagation-capable
            function, but ANY value. The higher the score, the better performing is the model.

            If no fitness function is passed, the negative loss of the model will be used as fitness.

        :param population:
            The number of individuals to be held in the generation.

        :param top_k:
            The size of the elite list to be selected based on the scores.

        :param mutation_rate:
            The percent of weights to be mutated in every child.

        :param mutation_std:
            The std from the normal distribution to be used for the sampling of mutations.

        :return:
            Generator yielding a tuple with [the generation (list of weights), max score, mean score, std score].
            Iterate over this generator to process each generation of the evolution.
        """
        def default_fitness_func(model, X, Y):
            result = model.evaluate(X, Y, batch_size=1024, verbose=0)

            if type(result) is list:
                result = result[0]

            result = result * -1
            return result

        if fitness_func is None:
            fitness_func = default_fitness_func

        self._adjust_generation(population)

        try:
            for generation_id in range(max_generations):
                best_weights = self._find_elite(fitness_func, X, Y, top_k=top_k)
                index = best_weights.index.to_series()

                max_score = np.round(index.max(), 4)
                mean_score = np.round(index.mean(), 4)
                std_score = np.round(index.std(), 4)

                self._generation = best_weights.tolist() + self._crossover(best_weights.tolist(), mutation_rate,
                                                                           mutation_std)
                self._model.set_weights(self._generation[0])

                tf.keras.backend.clear_session()
                yield self._generation, max_score, mean_score, std_score

        except KeyboardInterrupt as e:
            if len(self._generation) > 0:
                self._model.set_weights(self._generation[0])

            raise e from None