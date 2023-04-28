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




