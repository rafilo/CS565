from collections import defaultdict
import math
import random
import pandas as pd
import argparse
import statistics

"""
Accepts a list of points, each with the same number of dimensions.
(points can have more dimensions than 2)  
Returns a new point which is the center of all the points.
"""

def point_avg(points):
	center_point = statistics.median(points)
	return center_point

"""
Accepts a dataset and a list of assignments; the indexes 
of both lists correspond to each other.
Compute the center for each of the assigned groups.
Return `k` centers in a list
"""
def update_centers(data_set, assignments):
	centers = []
	assignment_dict = {}
	for i in range(len(assignment)):
		if i not in assignment_dict:
			assignment_dict[assignment[i]] = dataset[i]
		else:
			assignment_dict[assignment[i]] += dataset[i]
	
	for j in assignment_dict.values():
		centers.append(point_avg(j))
	return centers


def assign_points(data_points, centers):
    assignments = []
    for point in data_points:
        shortest = inf  # positive infinity
        shortest_index = 0
        for i in range(len(centers)):
            val = distance(point, centers[i])
            if val < shortest:
                shortest = val
                shortest_index = i
        assignments.append(shortest_index)
    return assignments


def distance(a, b):
	return abs(a-b)
    

"""
Given `data_set`, which is an array of arrays,
return a random set of k points from the data_set
"""
def generate_k(data_set, k):
    return random.sample(population=data_set, k=k)


def get_list_from_dataset_file(dataset_file):
	dataset = pd.read_csv(dataset_file)
	data = dataset[['revenue', 'vote_average']].astype(float).values.tolist()
	return data


def dim_1_k_means(dataset_file, k):
	tolerance = 0.001
	dataset = get_list_from_dataset_file(dataset_file)
	k_points = generate_k(dataset, k)
	assignments = assign_points(dataset, k_points)
	old_assignments = None
	
	while ((assignments - old_assignments)/old_assignments * 100) > tolerance:
        new_centers = update_centers(dataset, assignments)
    	old_assignments = assignments
    	assignments = assign_points(dataset, new_centers)
	clustering = defaultdict(list)
	for assignment, point in zip(assignments, dataset):
    	clustering[assignment].append(point)
	return clustering
