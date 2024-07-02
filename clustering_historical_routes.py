from sklearn.cluster import KMeans
import numpy as np
import dtw_similarity as dtws

# Extract historical routes
historical_routes = route_df['Route'].apply(lambda x: list(map(int, x.split(','))))

# Compute similarities
similarities = []
for route in historical_routes:
    similarity = dtws.compute_similarity(shortest_path, route)
    similarities.append(similarity)

# Cluster routes
kmeans = KMeans(n_clusters=10).fit(np.array(similarities).reshape(-1, 1))
route_df['Cluster'] = kmeans.labels_

# Select routes with 70-80% similarity
selected_routes = route_df[(route_df['Cluster'] == kmeans.predict([dtws.compute_similarity(shortest_path, shortest_path)])) & (np.array(similarities) >= 0.7) & (np.array(similarities) <= 0.8)]
