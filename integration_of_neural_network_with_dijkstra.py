import networkx as nx
import neural_network as nn


from collections import Counter

historical_routes = route_df['Route'].apply(lambda x: list(map(int, x.split(','))))

# Count edge usage
edge_counter = Counter()
for route in historical_routes:
    for edge in zip(route[:-1], route[1:]):
        edge_counter[edge] += 1

most_used_edges = edge_counter.most_common()



# Create graph
G = nx.DiGraph()

# Add edges and their attributes
for _, row in graph_df.iterrows():
    G.add_edge(row['NodeA'], row['NodeB'], weight=row['Weight'], edge_name=row['EdgeName'])

# Function to calculate duration
def calculate_duration(distance, speed):
    return distance / speed if speed > 0 else float('inf')

# Add durations to edges based on a sample timestamp
sample_instance = speed_df.iloc[0]  # Use the first instance as a sample
for edge in G.edges():
    speed = sample_instance[edge]
    distance = G[edge[0]][edge[1]]['weight']
    G[edge[0]][edge[1]]['duration'] = calculate_duration(distance, speed)

# Find shortest path
start_node = 94
end_node = 162
shortest_path = nx.dijkstra_path(G, start_node, end_node, weight='duration')

# Adjust weights based on neural network predictions and historical usage
def adjust_weights(G, historical_routes, most_used_edges, nn_model, timestamp):
    day, hour, minute, month = timestamp
    predicted_duration = nn.predict_duration(day, hour, minute, month)
    
    for edge, count in most_used_edges:
        G[edge[0]][edge[1]]['weight'] = G[edge[0]][edge[1]]['weight'] / predicted_duration
        if count > threshold:  # Define a threshold for "most used"
            G[edge[0]][edge[1]]['weight'] *= adjustment_factor  # Define an adjustment factor
    
    return G

# Adjust the graph weights
timestamp = (1, 12, 30, 'Apr')  # Example timestamp
G_adjusted = adjust_weights(G, historical_routes, most_used_edges, model, timestamp)

# Find the optimal route using the adjusted graph
optimal_route = nx.dijkstra_path(G_adjusted, start_node, end_node, weight='weight')
print("Optimal Route:", optimal_route)
