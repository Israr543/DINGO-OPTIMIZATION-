# DINGO-OPTIMIZATION-
import numpy as np

# Define the Sphere objective function (minimize sum of squares)
def sphere_function(x):
    return np.sum(x ** 2)

# Initialize population of agents (dingoes)
def initialize_population(pop_size, dim, bounds):
    lower, upper = bounds
    return np.random.uniform(lower, upper, (pop_size, dim))

# Clamp the agent within bounds
def clamp(agent, bounds):
    return np.clip(agent, bounds[0], bounds[1])

# Dingo Optimization Algorithm
def doa_optimize(obj_func, dim, bounds=(-10, 10), pop_size=30, max_iter=100):
    population = initialize_population(pop_size, dim, bounds)
    fitness = np.array([obj_func(ind) for ind in population])
    best_idx = np.argmin(fitness)
    best_agent = population[best_idx].copy()
    best_fitness = fitness[best_idx]

    for iteration in range(max_iter):
        for i in range(pop_size):
            strategy = np.random.choice(["group", "chase", "scavenge"])

            if strategy == "group":
                alpha = np.random.uniform(0.5, 1.0)
                new_agent = population[i] + alpha * (best_agent - population[i])
            elif strategy == "chase":
                j = np.random.randint(0, pop_size)
                beta = np.random.uniform(0.1, 0.5)
                new_agent = population[i] + beta * (population[j] - population[i])
            else:  # scavenge
                new_agent = initialize_population(1, dim, bounds)[0]

            # Clamp and evaluate
            new_agent = clamp(new_agent, bounds)
            new_fitness = obj_func(new_agent)

            # Survival rule: replace if better
            if new_fitness < fitness[i]:
                population[i] = new_agent
                fitness[i] = new_fitness
                if new_fitness < best_fitness:
                    best_fitness = new_fitness
                    best_agent = new_agent

        print(f"Iteration {iteration + 1}/{max_iter}, Best Fitness: {best_fitness:.5f}")

    return best_agent, best_fitness

# Main block
if __name__ == "__main__":
    dim = 30  # Number of dimensions
    best_sol, best_val = doa_optimize(sphere_function, dim)
    print("\nBest solution found:", best_sol)
    print("Best objective value:", best_val)
