from collections import Counter

historical_routes = route_df['Route'].apply(lambda x: list(map(int, x.split(','))))

# Count edge usage
edge_counter = Counter()
for route in historical_routes:
    for edge in zip(route[:-1], route[1:]):
        edge_counter[edge] += 1

most_used_edges = edge_counter.most_common()
