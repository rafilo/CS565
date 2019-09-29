from collections import defaultdict
from math import inf, sqrt
import random
import pandas as pd
import argparse
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

"""
read a csv file according to its' path, have to manualy choose dimention
"""
def get_list_from_dataset_file(dataset_file):
	dataset = pd.read_csv(dataset_file)
	data = dataset[['revenue', 'vote_average', 'popularity']].astype(float).values.tolist()
	idset = dataset['id']
	return data, idset

"""
cost function, not implemented yer
"""
def cost_function(clustering):
	tolerance = 0.001
	pass


"""
Accepts a list of points, each with the same number of dimensions.
Returns a new point which is the center of all the points.
"""

def point_avg(points):
	center_point = [0] * len(points[0])
	for curr_point in points:
		for j in range(len(curr_point)):
			center_point[j] += curr_point[j]
	for i in range(len(center_point)):
		center_point[i] = center_point[i] / len(points)

	return center_point

"""
Accepts a dataset and a list of assignments; the indexes 
of both lists correspond to each other.
recompute the center for each of the assigned groups.
Return updated centers in a list
"""
def update_centers(data_set, assignments):
	centers = []
	assignment_dict = defaultdict(list)
	for i in range(len(assignments)):
		if assignments[i] not in assignment_dict:
			assignment_dict[assignments[i]] = [data_set[i]]
		else:
			assignment_dict[assignments[i]] += [data_set[i]]
	for j in assignment_dict:
		centers.append(point_avg(assignment_dict[j]))

	return centers

"""
assign rest of points to its nearest cluster center
"""
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

"""
Returns the Euclidean distance between a and b (for dimention >= 2)
"""
def distance(a, b):
	squared_distance = 0
	for i in range(len(a)):
		squared_distance += (a[i] - b[i])**2
	return sqrt(squared_distance)

"""
return a random set of k points from the data_set
"""
def generate_k(data_set, k):
    return random.sample(population=data_set, k=k)

# -------------------------------------------------------------
# following function is for k++

"""
choose the next point with max proportional probability in k-means++
"""
def max_prop_probability(init_point, res_points):
	pass

"""
Given `data_set`, which is an array of arrays,
return a set of k points from the data_set according to k++ rule
"""
def generate_kpp(data_set, k):
    # need to modify 
	init_k = random.sample(population=data_set, k=1)
	iteration = k-1
	while iteration > 0:
		return 1

# -------------------------------------------------------------
# following function is for 1d kmeans
"""
returns the absolute distance between a and b for 1-dimentional clustering
"""
def d1_distance(a, b):
	return abs(a-b)


# -------------------------------------------------------------
# k_means main algorithm
def k_means(dataset_file, k, init):
	"""
	1. generate k centers according to the rule (k-means, k-means++, 1d)
	2. assign points to its nearest center (track old centers to determine the loss)
	3. loop until centers don't change anymore (or use lost function)
	4. output .csv file with movie's id and label
	"""
	dataset, idset = get_list_from_dataset_file(dataset_file)

	if init == 'random':
		k_points = generate_k(dataset, k)
	elif init == 'k-means++':
		# need to implement
		pass
	elif init == '1d':
		# need to implement
		pass

	assignments = assign_points(dataset, k_points)
	old_assignments = [0] * len(assignments)
	final_centers = []

	# need to add threshold
	while assignments != old_assignments:
		new_centers = update_centers(dataset, assignments)
		old_assignments = assignments
		assignments = assign_points(dataset, new_centers)
		final_centers = new_centers

	# assign group after threshold is met
	clustering = defaultdict(list)
	for assignment, point in zip(assignments, dataset):
		clustering[assignment].append(point)
	
	# outputs
	final_df = pd.DataFrame({'id': idset, 'label': assignments})
	final_df.to_csv('output.csv', index=False, header=True)
	return final_df

parser = argparse.ArgumentParser()
#parser.add_argument()
k_means('./movie.csv', 5, 'random')


