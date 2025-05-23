from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from src.problems.test_case_problem import TestCasePrioritizationProblem
from src.samplings.test_case_sampling import RandomFeasibleSampling
from src.common.test_crossover import TestCaseCrossover
from src.common.test_mutation import TestCaseMutation
from src.common.data_extraction import extract_data
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import time
import numpy as np


def random_search(test_data, n_samples, cost_time=60):

    start_time = time.time()
    fitness_values = []
    while time.time() - start_time < cost_time:

        random_selection = random.sample(test_data, n_samples)
        random_fitness = [item[1] for item in random_selection]
        mean_fitness = np.mean(random_fitness)
        fitness_values.append(mean_fitness)



    return fitness_values




if __name__ == "__main__":
   
    file_path = "test-case-prioritization/data/01-07-2024-all_tests_ga.json"
    with open(file_path, "r") as f:
        test_data = json.load(f)
    
    test_data, test_ids, test_objects = extract_data(test_data)
    #print("tests extraits :", test_data)    
    problem = TestCasePrioritizationProblem(test_data)
    sampling = RandomFeasibleSampling(test_data)
    pop_size = 2

    # Configurer l'algorithme
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=sampling,
        crossover=TestCaseCrossover(),
        mutation=TestCaseMutation(),
        eliminate_duplicates=True
    )

    
    cost_time = 300
    ga_time= int(cost_time/60)
    res_nsga2 = minimize(problem, algorithm, termination=("time", f"00:0{ga_time}:00"), seed=1, verbose=False)
    #random_fitness = random_search(problem, pop_size)
    # Sélection aléatoire de 100 éléments
    #random_selection = random.sample(test_data, pop_size)
    #random_fitness = [item[1] for item in random_selection]
    #print("Random Search Fitness:", random_fitness)
    
    # Collecter les fitness
    nsga2_fitness = res_nsga2.F[:, 0]  # Premier objectif de NSGA-II
    #print("nsga2 solution:", res_nsga2.X)
    #random_fitness = random_fitness[:, 0]  # Premier objectif de la recherche aléatoire
    nsga2_fitness = np.abs(nsga2_fitness)
    n = 10
    selected_indices = []
    # Sélectionner les indices des 10 meilleurs tests
    i=0
    for test_case in res_nsga2.X:
        if(len(selected_indices) < n):
            selected_indices.append(test_case)
        else:
            break
        #print("Test sélectionné :", test_case)
        # Vérifier si le test est déjà sélectionné       
    #print("Indices des tests sélectionnés :", selected_indices)
    selected_tests_data = {}
    for test_case  in selected_indices:
        for idx in test_case:
            run_key, test_id = test_ids[idx]
            if run_key not in selected_tests_data:
                selected_tests_data[run_key] = {}
            selected_tests_data[run_key][test_id] = test_objects[(run_key, test_id)]
    #print("Tests sélectionnés pour maximiser la diversité :")
    #print(selected_tests_data)
    print("******************* STARTING RANDOM SEARCH *******************")
    random_fitness = random_search(test_data = test_data, n_samples=pop_size, cost_time=cost_time)
    random_fitness = np.abs(random_fitness)
    # Afficher les résultats
    #print("NSGA-II Best Fitness:", np.min(nsga2_fitness))
    #print("Random Search Best Fitness:", np.min(random_fitness))

    # Comparer avec des visualisations
    data = {
        "NSGA-II": nsga2_fitness,
        "Random Search": random_fitness
    }
    
    df = pd.DataFrame({
    "Fitness": np.concatenate([nsga2_fitness, random_fitness]),
    "Method": ["NSGA-II"] * len(nsga2_fitness) + ["Random Search"] * len(random_fitness)
    })
    
    # Boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Method", y="Fitness", data=df)
    plt.title("Comparison of Fitness: NSGA-II vs Random Search (Boxplot)")
    plt.ylabel("Fitness")
    plt.xlabel("Method")
    plt.savefig("boxplot_comparison.png", dpi=300)
    plt.show()

    # Violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(x="Method", y="Fitness", data=df, inner="quartile")
    plt.title("Comparison of Fitness: NSGA-II vs Random Search (Violin Plot)")
    plt.ylabel("Fitness")
    plt.savefig("violinplot_comparison.png", dpi=300)
    plt.xlabel("Method")
    plt.show()
