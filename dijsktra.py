import networkx as nx

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
