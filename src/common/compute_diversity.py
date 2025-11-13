from scipy.spatial.distance import cosine as cosine_distances
import numpy as np
from .data_extraction import extract_features



"""
def extract_features(test_data):
    
    Transforme un test en un vecteur exploitable pour la distance cosinus.
    On utilise les positions et rotations des objets du test.
    
    vector = []
    for obj in test_data: 
        position = obj[0] 
        rotation = obj[1] 
        vector.extend(position + rotation) 
    #print("Vecteur extrait :", vector)
    return np.array(vector)
"""

def compute_diversity(candidate_order):
    """
     Calculates the fitness of a candidate solution (a permutation of test cases) 
     using the cosine distance between the features of consecutive test cases.
     
     f(candidate) = (1 / n) * somme_{i=2}^{n} (cosine_distance(test_i, test_{i-1}) / i)
     
    where n is the total number of test cases in the permutation.
    """
    n = len(candidate_order)
    total = 0.0
    for i in range(1, n):
        test_vector = [item[0] for item in candidate_order]
        test_i = test_vector[i]
        test_i_1 = test_vector[i - 1]
        
        #cosine distance between ti and ti-1
        distance = cosine_distances(test_i, test_i_1)
        total += distance / i
    #mAd = total / n
    return total
