import networkx as nx
import torch
import torch_geometric
import matplotlib.pyplot as plt

G = nx.karate_club_graph()
type(G)

G.is_directed()

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


#Question 3: What is the PageRank value for node 0 (node with id 0) after one PageRank iteration?

def one_iter_pagerank(G, b, r0, node_id):
    #create out variable of interest
    r1 = 0

    #create a map to store our info
    node_neigh = {}
    #for node_id, extract neighbours nodes and their degrees
    for node in G.neighbors(node_id):
        node_neigh[node] = G.degree[node]
    #first part of the pagerank equation
    for k, v in node_neigh:
        r1 += b * (r0/ v)
    #second part of the pagerank equation

    r1 += (1-b)* (1/G.number_of_nodes())

    return r1

beta = 0.8
r0 = 1.0 / G.number_of_nodes()
node = 0
r1 = one_iter_pagerank(G, beta, r0, node)
print("The PageRank value for node 0 after one iteration is {}".format(r1))

#Question 4: What is the (raw) closeness centrality for the karate club network node 5?

def closeness_centrality(G, n=5):
    shortest = nx.shortest_path_length(G, 5)
    shortest_sum = sum(shortest.values())
    #calculate raw closennes
    if shortest_sum ==0:
        closeness = 0
    else:
        closeness = (len(shortest) - 1) / shortest_sum

    closeness = round(closeness, 2)
    return closeness


node = 5
closeness = closeness_centrality(G, n=node)
print("The node 5 has closeness centrality {}".format(closeness))

#convert karateclub graph in a directed one
G_direc = nx.DiGraph(G)
G_direc.is_directed()


###### 2 PART, graphs to tensors
import torch
print(torch.__version__)


#Question 5: Get the edge list of the karate club network and transform
# it into torch.LongTensor. What is the torch.sum value of pos_edge_index tensor?

def graph_to_edge_list(G):
  # ODO: Implement the function that returns the edge list of
  # an nx.Graph. The returned edge_list should be a list of tuples
  # where each tuple is a tuple representing an edge connected
  # by two nodes.

  edge_list = []
  ############# Your code here ############
  edge_list = list(nx.edges(G))
  # ed_l = list(G.edges())
  #########################################
  return edge_list

def edge_list_to_tensor(edge_list):
  # ODO: Implement the function that transforms the edge_list to
  # tensor. The input edge_list is a list of tuples and the resulting
  # tensor should have the shape [2 x len(edge_list)].

  edge_index = torch.tensor([])
  ############# Your code here ############
  edge_index = torch.tensor(edge_list,dtype=torch.long).permute(1,0)
  #########################################
  return edge_index

pos_edge_list = graph_to_edge_list(G)
pos_edge_index = edge_list_to_tensor(pos_edge_list)
print("The pos_edge_index tensor has shape {}".format(pos_edge_index.shape))
print("The pos_edge_index tensor has sum value {}".format(torch.sum(pos_edge_index)))


edge_index = edge_list.type(torch.long)






