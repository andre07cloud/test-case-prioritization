import math
import numpy as np

def eucludien_distance(x, y):
    return math.sqrt(sum([(x[i] - y[i]) ** 2 for i in range(len(x))]))

def manhattan_distance(x, y):
    return sum([abs(x[i] - y[i]) for i in range(len(x))])

def minkowski_distance(x, y, p):
    return sum([abs(x[i] - y[i]) ** p for i in range(len(x))]) ** (1 / p)


def test_min_distance(test_data):
    boxes = test_data["test"][0]
    min_distance = float("inf")
    distances = []
    for i in range(len(boxes)):
        pos_i = boxes[i][0]
        for j in range(i + 1, len(boxes)):
            pos_j = boxes[j][0]
            distance = eucludien_distance(pos_i, pos_j)
            distances.append(distance)
            if distance < min_distance:
                min_distance = distance
    return min_distance, np.median(distances), distances
   