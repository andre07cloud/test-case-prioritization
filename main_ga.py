from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from src.problems.test_case_problem import TestCasePrioritizationProblem
from src.samplings.test_case_sampling import RandomFeasibleSampling
from src.common.test_crossover import TestCaseCrossover
from src.common.test_mutation import TestCaseMutation
from src.common.data_extraction import extract_data
from src.common.compute_diversity import compute_diversity
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import time



def random_search(test_data, n_samples, cost_time=60):

    start_time = time.time()
    fitness_values = []
    diversity_values = []
    while time.time() - start_time < cost_time:

        random_selection = random.sample(test_data, n_samples)
        diversity = compute_diversity(random_selection)
        #print("****** Diversity:", diversity)
        diversity_values.append(diversity)
        random_fitness = [item[1] for item in random_selection]
        mean_fitness = np.mean(random_fitness)
        fitness_values.append(mean_fitness)

    print("Random Search Done...:", )

    return fitness_values, diversity_values




if __name__ == "__main__":
   
    file_path = "data/01-07-2024-all_tests_ga.json"
    with open(file_path, "r") as f:
        test_data = json.load(f)
    
    test_data, test_ids, test_objects = extract_data(test_data)
    print("Probem Initialization...")    
    problem = TestCasePrioritizationProblem(test_data)
    print("Sampling Initialization...")
    sampling = RandomFeasibleSampling(test_data)
    pop_size = 100

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
    print("Starting NSGA-II optimization...")
    res_nsga2 = minimize(problem, algorithm, termination=("time", f"00:0{ga_time}:00"), seed=1, verbose=False)
    #random_fitness = random_search(problem, pop_size)
    # Sélection aléatoire de 100 éléments
    #random_selection = random.sample(test_data, pop_size)
    #random_fitness = [item[1] for item in random_selection]
    #print("Random Search Fitness:", random_fitness)
    
    # Collecter les fitness
    nsga2_fitness = res_nsga2.F[:, 0]  # Premier objectif de NSGA-II
    nsga2_diversity = res_nsga2.F[:, 1]  # Deuxième objectif de NSGA-II
    print("Nsga2_diversity:", nsga2_diversity)
    #print("nsga2 solution:", res_nsga2.X)
    #random_fitness = random_fitness[:, 0]  # Premier objectif de la recherche aléatoire
    nsga2_fitness = np.abs(nsga2_fitness)
    nsga2_diversity = np.abs(nsga2_diversity)
    print("Nsga2_diversity:", nsga2_diversity)
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
    random_fitness, random_diversity = random_search(test_data = test_data, n_samples=pop_size, cost_time=cost_time)
    #random_fitness = np.abs(random_fitness)
    #print("Random Search Done...:", random_diversity)
    random_diversity = np.abs(random_diversity)
    print("Random_Diversity:", random_diversity)
    # Afficher les résultats
    #print("NSGA-II Best Fitness:", np.min(nsga2_fitness))
    #print("Random Search Best Fitness:", np.min(random_fitness))

    # Comparer avec des visualisations
    data_difficulty = {
        "Optimized Prioritizer": nsga2_fitness,
        "Random Search": random_fitness
    }

    data_diversity = {
        "Optimized Prioritizer": nsga2_diversity,
        "Random Search": random_diversity
    }
    
    df_fitness = pd.DataFrame({
    "Difficulty": np.concatenate([nsga2_fitness, random_fitness]),
    "Method": ["Optimized Prioritizer"] * len(nsga2_fitness) + ["Random Search"] * len(random_fitness)
    })

    df_diversity = pd.DataFrame({
    "Diversity": np.concatenate([nsga2_diversity, random_diversity]),
    "Method": ["Optimized Prioritizer"] * len(nsga2_diversity) + ["Random Search"] * len(random_diversity)
    })
    
    # # Fitness Boxplot
    # plt.figure(figsize=(10, 6))
    # sns.boxplot(x="Method", y="Fitness", data=df)
    # plt.title("Comparison of Fitness: Optimized Method vs Random Search (Boxplot)")
    # plt.ylabel("Fitness")
    # plt.xlabel("Method")
    # plt.savefig("boxplot_comparison.png", dpi=300)
    # plt.show()

    # # Diversity Boxplot
    # plt.figure(figsize=(10, 6))
    # sns.boxplot(x="Method", y="Diversity", data=df_diversity)
    # plt.title("Comparison of Diversity: Optimized Method vs Random Search (Boxplot)")
    # plt.ylabel("Diversity")
    # plt.xlabel("Method")
    # plt.savefig("diversity_boxplot_comparison.png", dpi=300)
    # plt.show()

    # # Fitness Violin plot
    # plt.figure(figsize=(10, 6))
    # sns.violinplot(x="Method", y="Fitness", data=df, inner="quartile")
    # plt.title("Comparison of Fitness: Optimized Method vs Random Search (Violin Plot)")
    # plt.ylabel("Fitness")
    # plt.savefig("violinplot_comparison.png", dpi=300)
    # plt.xlabel("Method")
    # plt.show()

    # # Diversity Violin plot
    #     # Violin plot
    # plt.figure(figsize=(10, 6))
    # sns.violinplot(x="Method", y="Diversity", data=df_diversity, inner="quartile")
    # plt.title("Comparison of Diversity: Optimized Method vs Random Search (Violin Plot)")
    # plt.ylabel("Fitness")
    # plt.savefig("diversity_violinplot_comparison.png", dpi=300)
    # plt.xlabel("Method")
    # plt.show()

    # ax =sns.violinplot(data = df_fitness, palette='turbo', inner='box', linewidth=0, saturation=0.4, density_norm='width')
    # sns.boxplot(data = df_fitness, palette='turbo', width=0.2, boxprops = {'zorder': 2}, ax=ax)
    # sns.stripplot(data = df_fitness, palette='turbo', size=5, color='black', jitter=0.35, zorder=1, alpha=1, linewidth=1, edgecolor='black', ax=ax)
    # plt.xticks(ticks=[0, 1], labels=['Optimized Method', 'Random Search'])
    # plt.title("Comparison of Fitness: Optimized Method vs Random Search")
    # plt.ylabel("Fitness")
    # plt.xlabel("Method")
    # plt.savefig("combined_fitness_comparison.png", dpi=300)
    # plt.close()

    # ax =sns.violinplot(data = df_diversity, palette='turbo', inner='box', linewidth=0, saturation=0.4, density_norm='width')
    # sns.boxplot(data = df_diversity, palette='turbo', width=0.2, boxprops = {'zorder': 2}, ax=ax)
    # sns.stripplot(data = df_diversity, palette='turbo', size=5, color='black', jitter=0.35, zorder=1, alpha=1, linewidth=1, edgecolor='black', ax=ax)
    # plt.xticks(ticks=[0, 1], labels=['Optimized Method', 'Random Search'])
    # plt.title("Comparison of Fitness: Optimized Method vs Random Search")
    # plt.ylabel("Diversity")
    # plt.xlabel("Method")
    # plt.savefig("combined_Diversity_comparison.png", dpi=300)
    # plt.close()

# Fitness plot
# plt.figure(figsize=(10, 6))
# ax = sns.violinplot(x="Method", y="Fitness", data=df_fitness, color='skyblue', palette='turbo', inner='box', linewidth=0, saturation=0.4, density_norm='width')
# sns.boxplot(x="Method", y="Fitness", data=df_fitness, palette='turbo', width=0.2, boxprops={'zorder': 2, 'facecolor': 'lightblue', 'edgecolor': 'black'}, ax=ax)
# sns.stripplot(x="Method", y="Fitness", data=df_fitness, color='black', size=5, jitter=0.35, zorder=1, alpha=1, linewidth=1, ax=ax)
# plt.title("Comparison of Fitness: Optimized Method vs Random Search")
# plt.ylabel("Fitness")
# plt.xlabel("Method")

# Fitness plot
plt.figure(figsize=(10, 6))
ax = sns.violinplot(data=df_fitness, x='Method', y='Difficulty', color='skyblue', inner='box', palette={'Optimized Prioritizer': 'skyblue', 'Random Search': 'lightgreen'},)
sns.boxplot(x="Method", y="Difficulty", data=df_fitness, palette={'Optimized Prioritizer': 'skyblue', 'Random Search': 'lightgreen'}, 
    width=0.2, boxprops={'zorder': 2, 'facecolor': 'lightblue', 'edgecolor': 'black'},     whiskerprops={'color': 'black'},
    medianprops={'color': 'black', 'linewidth': 2}, showfliers=False, ax=ax)
plt.title('Violin Plot with Boxplot: Difficulty Perception Comparison (Optimized Prioritizer vs Random Search)', fontsize=14)
plt.xlabel('Method', fontsize=12)
plt.ylabel('Difficulty Score', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig("combined_fitness_comparison.png", dpi=300)
plt.close()

# Diversity plot
plt.figure(figsize=(10, 6))
ax = sns.violinplot(data=df_diversity, x='Method', y='Diversity', color='skyblue', inner='box', palette={'Optimized Prioritizer': 'skyblue', 'Random Search': 'lightgreen'},)
sns.boxplot(x="Method", y="Diversity", data=df_diversity, palette={'Optimized Prioritizer': 'skyblue', 'Random Search': 'lightgreen'}, 
    width=0.2, boxprops={'zorder': 2, 'facecolor': 'lightblue', 'edgecolor': 'black'},     whiskerprops={'color': 'black'},
    medianprops={'color': 'black', 'linewidth': 2}, showfliers=False, ax=ax)
plt.title('Violin Plot with Boxplot: Diversity Comparison (Optimized Prioritizer vs Random Search)', fontsize=14)
plt.xlabel('Method', fontsize=12)
plt.ylabel('Diversity Score', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig("combined_diversity_comparison.png", dpi=300)
plt.close()

# # Diversity plot
# plt.figure(figsize=(10, 6))
# ax = sns.violinplot(x="Method", y="Diversity", data=df_diversity, palette='turbo', inner=None, linewidth=0, saturation=0.4, density_norm='width')
# sns.boxplot(x="Method", y="Diversity", data=df_diversity, palette='turbo', width=0.2, boxprops={'zorder': 2}, ax=ax)
# sns.stripplot(x="Method", y="Diversity", data=df_diversity, color='black', size=5, jitter=0.35, zorder=1, alpha=1, linewidth=1, ax=ax)
# plt.title("Comparison of Diversity: Optimized Method vs Random Search")
# plt.ylabel("Diversity")
# plt.xlabel("Method")
# plt.savefig("combined_Diversity_comparison.png", dpi=300)
# plt.close()