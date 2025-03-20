import numpy as np


def test_max_deviation(test_data, reference_angle):
    boxes = test_data["test"][0]
    # Find the maximum deviation from 90 degrees for each box and the maximum deviation among all boxes
    print("*************** BOX i", boxes[2][1][2])
    deviations = [abs(reference_angle - boxes[i][1][2]) for i in range(len(boxes))]
    deviations = [abs(deviations[i] - np.mean(deviations)) / np.std(deviations) for i in range(len(deviations))]
    print("*************** Median BOX i", np.median(deviations))
    max_deviation = max(deviations)
   
    return max_deviation, deviations, np.median(deviations)