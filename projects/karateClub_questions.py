import networkx as nx
import torch
import torch_geometric
import matplotlib.pyplot as plt

G = nx.karate_club_graph()
type(G)

nx.draw_networkx(G, with_labels=True)
plt.show()

#Question 1: What is the average degree of the karate club network?
def average_degree(graph):
    degrees = G.degree()
    total_degree = 0
    for node, degree in degrees:
        total_degree += degree

    avg = total_degree / len(G.nodes)
    return avg

average_degree(G)

#Question 2: What is the average clustering coefficient of the karate club network?

noG = nx.Graph()
noG.add_nodes_from(range(1, 9))
noG.add_edges_from([(1, 2), (1, 3), (2, 3), (3, 4), (3, 5), (4, 5)])
noG.add_edges_from([(6, 7), (6, 8), (7, 8)])

def average_clustering_coefficient(G):
    if not nx.is_connected(G):
        print("Warning: Graph is not connected. Computing average clustering coefficient for each connected component.")
        ccs = list(nx.connected_components(G))
        avg_cc = sum(nx.average_clustering(G.subgraph(cc)) for cc in ccs) / len(ccs)
    else:
        avg_cc = nx.average_clustering(G)
    return avg_cc

average_clustering_coefficient(G)
average_clustering_coefficient(noG)






