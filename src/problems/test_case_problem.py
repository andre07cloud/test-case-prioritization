from pymoo.core.problem import Problem
#from src.common.vision_complexity import fitness_vison_complexity
from src.common.compute_diversity import compute_diversity
import numpy as np
#from src.common.data_extraction import extract_features

class TestCasePrioritizationProblem(Problem):
    def __init__(self, test_cases_data):
        # n_var : umber of variables describing a test case
        
        self.test_cases_data = test_cases_data  # List of all test cases
        n_test = len(test_cases_data)
        # Each solution is a permutation of indices from 0 to n_test-1
        xl = np.zeros(n_test)
        xu = np.array([n_test - 1] * n_test)
        super().__init__(n_var=n_test, n_obj=2, n_constr=0, xl=xl, xu=xu)
        

    def _evaluate(self, X, out, *args, **kwargs):
        f1_list, f2_list = [], []
        for candidate in X:
            
            candidate_order = [self.test_cases_data[i] for i in candidate]
            #print("Candidate order len:", len(candidate_order))
            print("Candidate:", candidate)
            #vision_complexity = - fitness_vison_complexity(candidate_order)
            test_fitness = [item[1] for item in candidate_order]
            fit1 = np.mean(test_fitness)
            #print("Test fitness:", test_fitness)
            #fit1 = test_fitness
            diversity = - compute_diversity(candidate_order)
            f1_list.append(fit1)
            f2_list.append(diversity)
            
        #fit3 = -np.mean(f1_list)
        out["F"] = np.column_stack([np.array(f1_list), np.array(f2_list)])
