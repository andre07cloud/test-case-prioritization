

import json

import numpy as np
import yaml
import pandas as pd
from typing import List, Tuple
import math
BOX_WIDTH = 0.14 # x 0.13205
BOX_HEIGHT= 0.1789 #_y 0.16495

def filter_duplicates_3d(failed_picks_locs):
    
    '''Function that filters out the duplicates in the failed pick locations'''

    filtered_failed_picks_locs = []
    for i, pick in enumerate(failed_picks_locs):
        if i == 0:
            filtered_failed_picks_locs.append(pick)
        else:
            if pick[0] != failed_picks_locs[i-1][0] or pick[1] != failed_picks_locs[i-1][1]:
                filtered_failed_picks_locs.append(pick)

    return filtered_failed_picks_locs

def filter_values_1d(values):
    
    '''Function that filters out the duplicates in the failed pick locations'''

    filtered_values = []
    for i, value in enumerate(values):
        if i == 0:
            filtered_values.append(value)
        else:
            if value != values[i-1]:
                filtered_values.append(value)

    return filtered_values

def get_rotations_from_location(failed_picks_locs):

    '''Function that finds the closest points to the given points
    and returns the corresponding rotations'''



    x1, y1 = test[0][0][0][:-1]
    x2, y2 = test[0][1][0][:-1]
    x3, y3 = test[0][2][0][:-1]

    points = [[x1, y1], [x2, y2], [x3, y3]]

    rotations = [(90 -test[0][0][1][2]), (90 - test[0][1][1][2]), (90 - test[0][2][1][2])]

    closest_rotations = []

    for point in failed_picks_locs:
        min_distance = float('inf')
        for i, other_point in enumerate(points):
            distance = np.linalg.norm(np.array(point[:2]) - np.array(other_point))
            if distance < min_distance:
                min_distance = distance
                closest_rotation = rotations[i]
        closest_rotations.append(closest_rotation)

    return closest_rotations

    



def distance_between_points(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def closest_neighbor_distances(points):
    distances = []
    for i, point in enumerate(points):
        min_distance = float('inf')
        for j, other_point in enumerate(points):
            if i != j:
                distance = distance_between_points(point, other_point)
                if distance < min_distance:
                    min_distance = distance
        distances.append(min_distance)
    return distances



def get_inter_box_closest_dist(test):

    x1, y1 = test[0][0][0][:-1]
    x2, y2 = test[0][1][0][:-1]
    x3, y3 = test[0][2][0][:-1]

    points = [[x1, y1], [x2, y2], [x3, y3]]

    distances = closest_neighbor_distances(points)


    return distances


def closest_pair_of_points_rotations(test):

    x1, y1 = test[0][0][0][:-1]
    x2, y2 = test[0][1][0][:-1]
    x3, y3 = test[0][2][0][:-1]

    points = [[x1, y1], [x2, y2], [x3, y3]]
    rotations = [abs(90 -test[0][0][1][2]), abs(90 - test[0][1][1][2]), abs(90 - test[0][2][1][2])]
    # Initialize variables to store the minimum distance and the pair of points
    min_distance = float('inf')
    closest_pair = None
    closest_rotations = None

    # Get the number of points
    n = len(points)

    # Loop through each pair of points
    for i in range(n):
        for j in range(i + 1, n):
            # Calculate the Euclidean distance between points[i] and points[j]
            distance = math.sqrt((points[i][0] - points[j][0])**2 + (points[i][1] - points[j][1])**2)
            
            # Update the minimum distance and closest pair if necessary
            if distance < min_distance:
                min_distance = distance
                closest_pair = (points[i], points[j])
                closest_rotations = (rotations[i], rotations[j])

    return closest_pair, closest_rotations


def get_inter_box_dist(test):

    x1, y1 = test[0][0][0][:-1]
    x2, y2 = test[0][1][0][:-1]
    x3, y3 = test[0][2][0][:-1]

    points = [[x1, y1], [x2, y2], [x3, y3]]

    distances = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distance = np.linalg.norm(np.array(points[i]) - np.array(points[j]))
            distances.append(distance)

    return distances

def get_inter_box_dist_given_box(point, test):


    x1, y1 = test[0][0][0][:-1]
    x2, y2 = test[0][1][0][:-1]
    x3, y3 = test[0][2][0][:-1]

    points = [[x1, y1], [x2, y2], [x3, y3]]


    distances = []
    for i in range(len(points)):
        distance = np.linalg.norm(np.array(point) - np.array(points[i]))
        distances.append(distance)

    distances = sorted(distances)


    return distances[1]

def get_robot_dist_failed(failed_picks_locs):


    if len(failed_picks_locs[0]) > 2:
        points = [p[:-1] for p in failed_picks_locs]
    else:
        points = [p for p in failed_picks_locs]
    robot_coord = np.array([-0.04, 0.257])

    distances = []
    for i in range(len(points)):
        distance = np.linalg.norm(np.array(points[i]) - robot_coord)
        distances.append(distance)

    return distances

def get_robot_dist(test):


    x1, y1 = test[0][0][0][:-1]
    x2, y2 = test[0][1][0][:-1]
    x3, y3 = test[0][2][0][:-1]

    robot_coord = np.array([-0.04, 0.257])

    points = [[x1, y1], [x2, y2], [x3, y3]]

    distances = []
    for i in range(len(points)):
        distance = np.linalg.norm(np.array(points[i]) - robot_coord)
        distances.append(distance)

    return distances

def get_x_y_list(test):

    x1, y1 = test[0][0][0][:-1]
    x2, y2 = test[0][1][0][:-1]
    x3, y3 = test[0][2][0][:-1]

    return np.array([x1, x2, x3]), np.array([y1, y2, y3])



def get_x_y_list_fail(positions):

    x = []
    y = []

    for pos in positions:
        x.append(pos[0])
        y.append(pos[1])


    return np.array(x), np.array(y)



if __name__ == "__main__":

    
    #repair = True

    
    '''
    uc = "uc2"
    
    test_file1 = "UC2\\repair_results\\06-07-2024_replay_results_random_box_use_case2_new_eval\\06-07-2024-all_tests.json"#"UC2\\initial_tests\\random\\05-07-2024_all_tests_random_box_use_case_parallel_new\\05-07-2024-all_tests.json"# "/home/ubuntu/docker/isaac-sim/documents/testing_intelligent_robotics/25-05-2024_replay_results_random_box_use_case2/25-05-2024-all_tests.json"#"/home/ubuntu/docker/isaac-sim/documents/testing_intelligent_robotics/25-05-2024_replay_results_random_box_use_case2/25-05-2024-all_tests.json"#  "/home/ubuntu/docker/isaac-sim/documents/testing_intelligent_robotics/01-03-2024_all_tests_random_box_use_case1_sunction2/01-03-2024-all_tests.json"
    test_file2 = "UC2\\repair_results\\06-07-2024_replay_results_ga_box_use_case2_new_eval\\06-07-2024-all_tests.json" #"UC2\\initial_tests\\ga\\01-07-2024_all_tests_ga_box_use_case_parallel_new\\01-07-2024-all_tests.json" ##  "/home/ubuntu/docker/isaac-sim/documents/testing_intelligent_robotics/25-05-2024_replay_results_ga_box_use_case2/25-05-2024-all_tests.json"#
    '''
    
    
    uc = "uc1"
    test_file1 = "UC1\\repair\\random\\08-07-2024_replay_results_random_box_use_case1_sunction_new_eval3\\08-07-2024-all_tests.json"#"UC1\\initial_tests\\random\\01-07-2024_all_tests_random_box_use_case1_sunction\\01-07-2024-all_tests.json" #"/home/ubuntu/docker/isaac-sim/documents/testing_intelligent_robotics/01-03-2024_all_tests_random_box_use_case1_sunction2/01-03-2024-all_tests.json"#"#"/home/ubuntu/docker/isaac-sim/documents/testing_intelligent_robotics/25-05-2024_replay_results_random_box_use_case1_sunction/25-05-2024-all_tests.json" #
    test_file2 = "UC1\\repair\\ga\\07-07-2024_replay_results_ga_box_use_case1_sunction_eval\\07-07-2024-all_tests.json"#"UC1\\initial_tests\\ga\\29-06-2024_all_tests_ga_box_use_case1_sunction\\29-06-2024-all_tests.json"# "/home/ubuntu/docker/isaac-sim/documents/testing_intelligent_robotics/08-03-2024_all_tests_ga_box_use_case1_sunction3/08-03-2024-all_tests.json"#"/home/ubuntu/docker/isaac-sim/documents/testing_intelligent_robotics/26-05-2024_replay_results_ga_box_use_case1_sunction/26-05-2024-all_tests.json" #
    
    with open(test_file1, "r") as f:
        all_tests1 = json.load(f)

    with open(test_file2, "r") as f:
        all_tests2 = json.load(f)

    all_test_names = [all_tests1, all_tests2]



    
    #feature_names = ["passed", "failed", "req1_failed", "req2_failed", "req1_req2_failed", "algo"] #, "inter_box_dist_rot" 

    column_names = ["x", "y", "rot", "inter_box_dist", "robot_dist", "algo", "label"] #, "inter_box_dist_rot"

    static_features_columns = ["luminosity", "algo", "label", "severity", "exec_time", "req1_failed", "req2_failed", "req1_req2_failed", "rot_deviation"]
    


    print("Column names", column_names)

    #features = []
   # operations = [np.mean, np.min, np.max] #, np.std

    df_box = []
    df_general = []

    a = 0



    for all_tests in all_test_names:

        print("Analysing files")

        for run in all_tests:
            for tc in all_tests[run]:
                req1_failed = 0
                req2_failed = 0
                req1_req2_failed = 0
                rot_deviation = 0
                if all_tests[run][tc]["info"] == "Valid test":
                    # analysing failing tests
                    if a == 0:
                        algo = "rs" #0 - fail random
                    else:
                        algo = "ga"# fail ga

                    if all_tests[run][tc]["test_outcome"] == "PASS" or all_tests[run][tc]["test_outcome"] == "TIMEOUT" or all_tests[run][tc]["test_outcome"] == "CRASH":

                        if all_tests[run][tc]["test_outcome"] == "PASS":
                            label = "pass"
                            rot_deviation = all_tests[run][tc]["max_rot_deviation"]
                        elif all_tests[run][tc]["test_outcome"] == "TIMEOUT":
                            label = "timeout"
                        elif all_tests[run][tc]["test_outcome"] == "CRASH":
                            label = "crash"
                        else:
                            print("Unknown test outcome")
                            exit(0)

                        test =  all_tests[run][tc]["test"]

                        severity = 0



                        exec_time = all_tests[run][tc]["time"]
                        lumosity = test[-1]
                        
                            
                        inter_box_dist = get_inter_box_closest_dist(test)
                        x, y = get_x_y_list(test)
                        rotations = [(90 -test[0][0][1][2]), (90 - test[0][1][1][2]), (90 - test[0][2][1][2])]
                        inter_box_dist_rot = np.array(inter_box_dist)*np.array(rotations)
                        robot_dist = get_robot_dist(test)

                        current_box_features = []
                        for i, item in enumerate(inter_box_dist):
                            current_box_features = [x[i],
                                                    y[i],
                                                    rotations[i],
                                                    inter_box_dist[i],
                                                    robot_dist[i],
                                                    algo,
                                                    label]

                            df_box.append(current_box_features)

                        df_general.append([lumosity, algo, label, severity, exec_time, req1_failed, req2_failed, req1_req2_failed, rot_deviation])

                        
                    elif all_tests[run][tc]["test_outcome"] == "FAIL": # or all_tests[run][tc]["test_outcome"] == "FAIL":
                        
                        test =  all_tests[run][tc]["test"]

                        exec_time = all_tests[run][tc]["time"]
                        lumosity = test[-1]


                        failure_codes = all_tests[run][tc]["failure_codes"]
                        max_pick_dist = all_tests[run][tc]["pick_distance"]
                        if max_pick_dist > 0.02:
                            print(f"{algo} {run} {tc}")
                            if not(2 in failure_codes):
                                failure_codes.append(2)

                        severity = len(failure_codes)

                        if (1 in failure_codes or 2 in failure_codes) and (not(4 in failure_codes) and not(5 in failure_codes)):
                            label = "dnn_fail"
                            rot_deviation = all_tests[run][tc]["max_rot_deviation"]
                        elif (4 in failure_codes or 5 in failure_codes) and (not(1 in failure_codes) and not(2 in failure_codes)):
                            label = "hard_fail"
                            if 4 in failure_codes and 5 in failure_codes:
                                req1_req2_failed = 1
                            elif 4 in failure_codes and 5 not in failure_codes:
                                req1_failed = 1
                            elif 5 in failure_codes and 4 not in failure_codes:
                                req2_failed = 1
                        elif (1 in failure_codes or 2 in failure_codes) and (4 in failure_codes or 5 in failure_codes):
                            label = "soft_fail"
                            rot_deviation = all_tests[run][tc]["max_rot_deviation"]
                            if 4 in failure_codes and 5 in failure_codes:
                                req1_req2_failed = 1
                            elif 4 in failure_codes and 5 not in failure_codes:
                                req1_failed = 1
                            elif 5 in failure_codes and 4 not in failure_codes:
                                req2_failed = 1
                        else:
                            print("Unknown failure code")
                            exit(0)

                         
                         

                        failed_picks_locs =  all_tests[run][tc]["failed_picks_locs"]

                        failed_picks_locs = filter_duplicates_3d(failed_picks_locs)

                        x, y = get_x_y_list_fail(failed_picks_locs)

                        if label == "dnn_fail":
                            rotations = all_tests[run][tc]["failed_rotations"]
                        elif label == "hard_fail" or label == "soft_fail":
                            rotations = get_rotations_from_location(failed_picks_locs)


                        inter_box_dist = all_tests[run][tc]["failed_box_distance"]
                        inter_box_dist = filter_values_1d(inter_box_dist)

                        robot_dist = all_tests[run][tc]["failed_distances"]
                        robot_dist = filter_values_1d(robot_dist)



                        current_box_features = []
                        for i, item in enumerate(inter_box_dist):
                            current_box_features = [x[i],
                                                    y[i],
                                                    rotations[i],
                                                    inter_box_dist[i],
                                                    robot_dist[i],
                                                    algo,
                                                    label ]

                            df_box.append(current_box_features)

                        df_general.append([lumosity, algo, label, severity, exec_time, req1_failed, req2_failed, req1_req2_failed, rot_deviation])

                    else:
                        print(f"Unknown test outcome {all_tests[run][tc]['test_outcome']}")
                        print("Unknown test outcome")
                        exit(0)




        a += 1


    #df1 = pd.DataFrame(df1, columns=column_names_1)
    df1 = pd.DataFrame(df_box, columns=column_names)

    df2 = pd.DataFrame(df_general, columns=static_features_columns)



    print("Saving files")

    # Save DataFrame to a CSV file
    #df1.to_csv('05-29-2024-uc2-dataset-tests.csv', index=False)
    # Save DataFrame to a CSV file
    df1.to_csv('UC1\\repair\\box_dataset_'+uc + '.csv', index=False)
    df2.to_csv('UC1\\repair\\box_dataset_general_'+uc+ '.csv', index=False)





