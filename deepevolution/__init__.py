__version__ = "0.0.4"


def wrap_keras():
    from tensorflow.keras.models import Model
    from deepevolution.deepevolution import DeepEvolution
    from tqdm.auto import tqdm

    def _evolve_wrapper(self, X, Y, max_generations=100, fitness_func=None, population=16, top_k=4, mutation_rate=0.2,
                        mutation_std=0.03):
        de = DeepEvolution(self)
        return de.evolve(X, Y, max_generations, fitness_func, population, top_k, mutation_rate, mutation_std)

    def _fit_evolve_wrapper(self, X, Y, max_generations=100, fitness_func=None, population=16, top_k=4,
                            mutation_rate=0.2, mutation_std=0.03, verbose=1):
        de = DeepEvolution(self)
        update_func = None

        if verbose == 1:
            update_func = lambda generation_id, mean_score, best_score, std_score: print(
                f"[Generation {generation_id} / {max_generations}] score: {mean_score} (best: {best_score}; std: {std_score})")
        elif verbose == 2:
            tqdm_verbose = tqdm(total=max_generations, dynamic_ncols=True)
            update_func = lambda generation_id, mean_score, best_score, std_score: tqdm_verbose.update(
                1) or tqdm_verbose.set_description(
                f"[Generation {generation_id} / {max_generations}] score: {mean_score} (best: {best_score}; std: {std_score})")

        history = {
            'score': []
        }

        for generation_id, (_, best_score, mean_score, std_score) in enumerate(
                de.evolve(X, Y, max_generations, fitness_func, population, top_k, mutation_rate, mutation_std)):
            generation_id += 1
            history['score'].append(mean_score)
            if update_func:
                update_func(generation_id, mean_score, best_score, std_score)

        return history

    Model.evolve = _evolve_wrapper
    Model.fit_evolve = _fit_evolve_wrapper
