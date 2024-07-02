from tslearn.metrics import dtw
from scipy.spatial.distance import euclidean

# Function to compute DTW similarity
def compute_similarity(route1, route2):
    return dtw(route1, route2)

# Function to compute edit distance similarity
def edit_distance(route1, route2):
    return sum(1 for a, b in zip(route1, route2) if a != b) + abs(len(route1) - len(route2))
