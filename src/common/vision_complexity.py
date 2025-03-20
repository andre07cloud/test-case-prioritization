import math
import numpy as np
from .calculate_center_deviation import test_max_center_deviation
from .calculate_max_rotation import test_max_deviation
from .calculate_distance import test_min_distance


def fitness_vison_complexity(test_data):
    center = [-0.7, 0.26, 0.786]
    reference_angle = 90
    max_center_distance, median_center_distance, center_distances = test_max_center_deviation(test_data, center)
    min_distance, median_distances, distances = test_min_distance(test_data)
    max_deviation, deviations, median_deviations = test_max_deviation(test_data, reference_angle)
    
    print("Max Center Distance: ", max_center_distance)
    print("Min Distance: ", min_distance)
    print("Max Deviation: ", max_deviation)
    alpha = 0.5
    beta = 0.3
    gamma = 0.2
    fitness_vision = alpha * max_center_distance + beta * min_distance + gamma *  max_deviation
    print("Fitness Vision:", fitness_vision)
    return fitness_vision
    