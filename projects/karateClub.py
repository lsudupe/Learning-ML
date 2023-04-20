from torch_geometric.datasets import KarateClub
import networkx as nx
from torch_geometric.utils import to_networkx



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
p


nx.draw(data, with_labels=True)