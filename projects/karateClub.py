from torch_geometric.datasets import KarateClub
import networkx as nx
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt




# check our data
dataset = KarateClub()
print(f"Dataset:{dataset}")
print("#################")
print(f'Number of graphs: {len(dataset)}')
print(f"Number of features: {dataset.num_features}")
print(f"Number of classes: {dataset.num_classes}")

# get the first graph object
data = dataset[0]
print(data)

# get some info from our object
print(f"n of nodes: {data.num_nodes}")
print(f"n of edges: {data.num_edges}")
print(f"average node degree: {2*data.num_edges / data.num_nodes}")
print(f"n of training nodes: {data.train_mask.sum()}")
print(f"training node label rate: {data.train_mask.sum() / data.num_nodes:.2f}, which is the percentage of labeled nodes in the trainning set")
print(f"contains isolated nodes: {data.has_isolated_nodes()}")
print(f"contains self-loops: {data.has_self_loops()}")
print(f"is undirected: {data.is_undirected()}")

nx.draw(data, with_labels=True)

#
data.edge_index.T #this representation is known as the COO format (coordinate format)
#commonly used to representing sparce matrices. Instead of holding the adjacency information
#in a dense representation, PyG represents graphs sparsely, only holding the coordinates/values
#for hich entries in A are non-zero

G = to_networkx(data, to_undirected=True)
nx.draw_networkx(G, node_color=data.y)
plt.show()
#plt.savefig("filename.png")


###Implementing Graph Neural Networks (GNNs)















