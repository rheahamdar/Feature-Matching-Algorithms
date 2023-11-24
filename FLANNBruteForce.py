import numpy as np

def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def brute_force_knn(query_point, data, k=1):
    distances = [euclidean_distance(query_point, data_point) for data_point in data]
    indices = np.argsort(distances)[:k] #to find the k nearest neighbor
    return indices

data = np.random.rand(100, 2)

query_point = np.array([0.5, 0.5])

# Number of nearest neighbors to find
k_neighbors = 3

nearest_indices = brute_force_knn(query_point, data, k_neighbors)

print("Query Point:", query_point)
print("Nearest Neighbors Indices:", nearest_indices)
print("Nearest Neighbors:", data[nearest_indices])