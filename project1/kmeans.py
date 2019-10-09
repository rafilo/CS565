from collections import defaultdict
from math import inf, sqrt
import random
import pandas as pd
import argparse
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def get_list_from_dataset_file_PCA(dataset_file):
    dataset = pd.read_csv(dataset_file)
    vote_avg = dataset['vote_average']
    vote_cnt = dataset['vote_count']
    total_vote = []
    for i in range(len(vote_avg)):
        total_vote.append(vote_avg[i] * vote_cnt[i])
    dataset['total_votes'] = total_vote
    sorted_df = dataset.sort_values('total_votes', ascending=False)
    top_df = sorted_df.head(250)
    final_df = top_df[['revenue', 'total_votes']].astype(float).values.tolist()
    idset = top_df['id']
    return final_df, idset

"""
read a csv file according to its' path, have to manualy choose dimention
"""
def get_list_from_dataset_file(dataset_file):
	dataset = pd.read_csv(dataset_file)
	data = dataset[['revenue', 'popularity','vote_average']].astype(float).values.tolist()
	idset = dataset['id']
	return data, idset


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
Returns the Euclidean distance between a and b (for dimention >= 2)
"""
def euclidean_distance(a, b):
	squared_distance = 0
	for i in range(len(a)):
		squared_distance += (a[i] - b[i])**2
	return sqrt(squared_distance)

"""
assign rest of points to its nearest cluster center
"""
def assign_points(data_points, centers, init):
	assignments = []
	for point in data_points:
		shortest = inf  # positive infinity
		shortest_index = 0
		for i in range(len(centers)):
			val = 0
			if init == '1d':
				val = d1_distance(point, centers[i])
			else:
				val = euclidean_distance(point, centers[i])

			if val < shortest:
				shortest = val
				shortest_index = i
		assignments.append(shortest_index)
	return assignments


"""
return a random set of k points from the data_set
"""
def generate_k(data_set, k):
    return random.sample(population=data_set, k=k) 

# -------------------------------------------------------------
# following function is for k++
def kpp_distance(a, b):
	squared_distance = 0
	for i in range(len(a)):
		squared_distance += (a[i] - b[i])**2
	return squared_distance
"""
choose the next point with max proportional probability in k-means++
"""
def choose_prop_probability(init_point, res_points):
	prop_dis_list = []
	prop_denominator = 0
	for i in res_points:
		prop_denominator += kpp_distance(init_point, i)

	for k in res_points:
		prop_dis_list.append((kpp_distance(init_point,k))/prop_denominator)
	return prop_dis_list

"""
Given `data_set`, which is an array of arrays,
return a set of k points from the data_set according to k++ rule
"""
def generate_kpp(data_set, k):
    # need to modify 
	"""
	choose other points proportional to the probability!!!
	"""
	init_k = random.sample(population=data_set, k=1)
	curr_k = init_k
	while k-1 > 0:
		next_k = random.choices(data_set, weights=choose_prop_probability(init_k[0], data_set))
		init_k.append(next_k[0])
		curr_k = next_k
		k -= 1
	return init_k

# -------------------------------------------------------------
# following function is for 1d kmeans
"""
returns the absolute distance between a and b for 1-dimentional clustering
"""
def d1_distance(a, b):
	d1_dist = 0
	for i in range(len(a)):
		d1_dist += abs(a[i]-b[i])
	return d1_dist

# -------------------------------------------------------------
# dynamic programming for 1d
"""
calculate the unit cost given the range of points
points: input data
i, j: indices
"""
def unit_cost(points, i, j):
	point_range = points[i:j]
	for a in range(len(point_range)):
		point_range[a] = point_range[a][0]
	rep = (1/(j-i+1)) * sum(point_range)
	cost = 0
	for point in point_range:
		cost += (point - rep) ** 2
	return cost

"""
run the dynamic programming
----------- 
X: set of 1-dimentional data
n: number of points
k: number of clustering
"""
def OPT(X, n, k):
    partition_indices = []
    opt = []
	# use for visualize
    result = []

	# initialize opt cost matrix / partition indices
    for i in range(k + 1):
        opt.append([0] * n)
        partition_indices.append([-1] * n)
        result.append([])

	# main part of dynamic programming
	# init unit_cost for 1st iter
    for i in range(n):                                     
        opt[1][i] = unit_cost(X, 0, i)

	# OPT(a,j) = min {OPT(a-1,j) + unit_cost(j+1, q)}
    for clusters in range(2, k + 1):
        for points in range (clusters - 1, n):
            min =  float('inf')
            partition_index = -1

            for j in range(clusters - 2, points):
                if opt[clusters - 1][j] + unit_cost(X, j + 1, points) < min:
                    min = opt[clusters - 1][j] + unit_cost(X, j + 1, points)
                    partition_index = j
            partition_indices[clusters][points] = partition_index
            opt[clusters][points] = min

    partition_indices = find_partition_indices(partition_indices, n, k)
    result = partition(X, partition_indices)
    return partition_indices

"""
find the partition indices according to the result of OPT
partition_indices: partition indices found in the OPT()
n: number of points
k: number of clusters
"""
def find_partition_indices(partition_indices, n, k):
    partition_index = [0] * (k - 1)
    partition_index[0] = partition_indices[k][n - 1]
    q = 0
	# back track
    for i in range(k-1, 1, -1):
        for j in range(n-1, i-2, -1):
            if j == partition_index[q]:
                partition_index[q+1] = partition_indices[i][j]
                q = q + 1
                break
    partition_index.reverse()

    return partition_index
"""
find the cluster that a point belongs to
X: input data
partition_indices
"""
def partition(X, partition_indices):
    result = [[]* (len(partition_indices))]
    data_label = []
    j = 0
    q = 0
    for i in range(len(X)):
        if q < len(partition_indices):
            if i <= partition_indices[q]:
                result[j].append(X[i])
            else:
                q = q + 1
                j = j + 1
                result[j].append(X[i])
        else:
            result[len(partition_indices)].append(X[i])
    for i in range(len(result)):
        for j in range (len(result[i])):
            data_clust = []
            data_clust.append(result[i][j])
            data_clust.append(i)
            data_label.append(data_clust)
    return data_label


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

	if init == '1d':
		regulized_data = StandardScaler().fit_transform(dataset)
		pca = PCA(n_components=1)
		dataset = pca.fit_transform(regulized_data)
		k_points = generate_k(list(dataset), k)
	if init == 'random':
		k_points = generate_k(dataset, k)
	elif init == 'k-means++':
		k_points = generate_kpp(dataset,k)

	assignments = assign_points(dataset, k_points, init)
	old_assignments = [0] * len(assignments)
	final_centers = []
	final_assignments = []
	# need to add threshold
	while assignments != old_assignments:
		new_centers = update_centers(dataset, assignments)
		old_assignments = assignments
		assignments = assign_points(dataset, new_centers, init)
		final_centers = new_centers
		final_assignments = assignments
	# assign group after threshold is met
	clustering = defaultdict(list)
	for assignment, point in zip(assignments, dataset):
		clustering[assignment].append(point)
	
	# outputs / need to move to cost_compare
	final_df = pd.DataFrame({'id': idset, 'label': assignments})
	final_df.to_csv('output.csv', index=False, header=True)
	return final_assignments, final_centers

def k_means_cost(clustering):
	if clustering is None or len(clustering) == 0:
		raise Exception("clustering should not be empty!")
	cost = 0
	
	for _, v in clustering.items():
		center = point_avg(v)
		for point in v:
			cost += euclidean_distance(center, point)**2
	return cost

# disagreement_dist
def disagreement_dist(assignment1, assignment2):
	dis_distance = 0
	for i in range(len(assignment1)):
		for j in range(i,len(assignment1)):
			if (assignment1[i] == assignment1[j]) and (assignment2[i] != assignment2[j]):
				dis_distance += 1
			elif (assignment1[i] != assignment1[j]) and (assignment2[i] == assignment2[j]):
				dis_distance += 1
	return dis_distance



parser = argparse.ArgumentParser()
parser.add_argument('filename', help='input file')
parser.add_argument('clusters', type=int,help='number of cluster', default=3)
parser.add_argument('init', help='type of kmeans', default='random')

#args = parser.parse_args()
#k_means(args.filename, args.clusters, args.init)

#kmeanspass = k_means('./movie.csv', 3, 'k-means++')
#kmeans1d = k_means('./movie.csv', 3, '1d')
dataset, idset = get_list_from_dataset_file('./movie.csv')
regulized_data = StandardScaler().fit_transform(dataset)
pca = PCA(n_components=1)
dataset = pca.fit_transform(regulized_data)
result = OPT(dataset, len(dataset), 3)
print(result)
"""
label1_x =[]
label1_y =[]

label2_x=[]
label2_y=[]

label3_x=[]
label3_y=[]

center_x =[]
center_y=[]
for a in centers:
	center_x.append(a[0])
	center_y.append(a[1])
for i in clst[0]:
	label1_x.append(i[0])
	label1_y.append(i[1])

for k in clst[1]:
	label2_x.append(k[0])
	label2_y.append(k[1])

for j in clst[2]:
	label3_x.append(j[0])
	label3_y.append(j[1])

plt.xlabel('total votes')
plt.ylabel('popularity')
plt.scatter(label1_x, label1_y, c='r', alpha=0.5)
plt.scatter(label2_x, label2_y, c='g', alpha=0.5)
plt.scatter(label3_x, label3_y, c='b', alpha=0.5)
plt.scatter(center_x, center_y, c='black', s=200)
plt.show()
"""
"""
_, labels = k_means('./movie.csv', 3, 'random')
data,_ = get_list_from_dataset_file_PCA('./movie.csv')
regulized_data = StandardScaler().fit_transform(data)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(regulized_data)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
df_labels = pd.DataFrame({'labels': labels})
finalDf = pd.concat([principalDf, df_labels], axis = 1)
plot_df = finalDf.astype(float).values.tolist()

label1_x =[]
label1_y =[]

label2_x=[]
label2_y=[]

label3_x=[]
label3_y=[]
for i in plot_df:
	if i[2] == 0.0:
		label1_x.append(i[0])
		label1_y.append(i[1])
	if i[2] == 1.0:
		label2_x.append(i[0])
		label2_y.append(i[1])
	if i[2] == 2.0:
		label3_x.append(i[0])
		label3_y.append(i[1])

plt.scatter(label1_x, label1_y, c='r', alpha=0.5, label='group1')
plt.scatter(label2_x, label2_y, c='g', alpha=0.5, label='group2')
plt.scatter(label3_x, label3_y, c='b', alpha=0.5, label='group3')
plt.show()
"""