import math
import numpy as np

def eucludien_distance(x, y):
    return math.sqrt(sum([(x[i] - y[i]) ** 2 for i in range(len(x))]))

def manhattan_distance(x, y):
    return sum([abs(x[i] - y[i]) for i in range(len(x))])

def minkowski_distance(x, y, p):
    return sum([abs(x[i] - y[i]) ** p for i in range(len(x))]) ** (1 / p)


def test_max_center_deviation(test_data, center):
    boxes = test_data["test"][0]
    max_deviation = -float("inf")
    deviations = []
    for i in range(len(boxes)):
        pos_i = boxes[i][0]
        deviation = eucludien_distance(pos_i, center)
        deviations.append(deviation)
        if deviation > max_deviation:
            max_deviation = deviation
    return max_deviation, np.median(deviations), deviations