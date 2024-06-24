import parse_csv as pc
import distance as dis


def dist_matrix():
    matrix = []
    parse = pc.parse_csv()

    for i in range(538):
        matrix.append([0 for x in range(538)])

    for le in range(1, len(parse)):
        f = parse[le].get('from')
        t = parse[le].get('to')
        matrix[int(f)][int(t)] = dis.get_edge_num_dist(int(parse[le].get('number')))

    return matrix
