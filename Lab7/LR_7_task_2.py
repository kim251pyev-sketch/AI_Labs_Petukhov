import numpy as np
import matplotlib.pyplot as plt
from deap import base, benchmarks, cma, creator, tools


def create_toolbox(strategy):
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("evaluate", benchmarks.rastrigin)
    return toolbox


if __name__ == "__main__":
    num_individuals = 10
    num_generations = 125

    strategy = cma.Strategy(centroid=[5.0] * num_individuals, sigma=5.0, lambda_=20 * num_individuals)
    toolbox = create_toolbox(strategy)

    np.random.seed(7)

    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    hall_of_fame = tools.HallOfFame(1)
    stats = tools.Statistics(lambda x: x.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    sigma = np.ndarray((num_generations, 1))
    axis_ratio = np.ndarray((num_generations, 1))
    diagD = np.ndarray((num_generations, num_individuals))
    fbest = np.ndarray((num_generations, 1))
    best = np.ndarray((num_generations, num_individuals))
    std = np.ndarray((num_generations, num_individuals))

    for gen in range(num_generations):
        population = toolbox.generate()
        fitnesses = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        toolbox.update(population)
        hall_of_fame.update(population)
        record = stats.compile(population)
        logbook.record(evals=len(population), gen=gen, **record)

        print(logbook.stream)

        sigma[gen] = strategy.sigma
        axis_ratio[gen] = max(strategy.diagD) ** 2 / min(strategy.diagD) ** 2
        diagD[gen, :] = strategy.diagD ** 2
        fbest[gen] = hall_of_fame[0].fitness.values
        best[gen, :] = hall_of_fame[0]
        std[gen, :] = np.std(population, axis=0)

    x = list(range(0, strategy.lambda_ * num_generations, strategy.lambda_))
    avg, max_, min_ = logbook.select("avg", "max", "min")

    plt.figure()
    plt.semilogy(x, avg, "--b", label="avg")
    plt.semilogy(x, max_, "--b", label="max")
    plt.semilogy(x, min_, "-b", label="min")
    plt.semilogy(x, fbest, "-c", label="fbest")
    plt.semilogy(x, sigma, "-g", label="sigma")
    plt.semilogy(x, axis_ratio, "-r", label="axis ratio")
    plt.grid(True)
    plt.title("Синій: f-значення, зелений: sigma, червоний: axis ratio")

    plt.figure();
    plt.plot(x, best);
    plt.grid(True);
    plt.title("Об'єктні змінні")
    plt.figure();
    plt.semilogy(x, diagD);
    plt.grid(True);
    plt.title("Масштабування")
    plt.figure();
    plt.semilogy(x, std);
    plt.grid(True);
    plt.title("Стандартне відхилення")
    plt.show()