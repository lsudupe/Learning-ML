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
import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1234)
        self.conv1 = GCNConv(data.num_features, 4) #it stored the output as a attribute of the GCNmodel
        self.conv2 = GCNConv(3, 3)
        self.conv3 = GCNConv(3, 2)
        self.classifier = Linear(2, len(data.keys))

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh() #Final GNN embedding space
        #Apply a final (linear) classifier
        out = self.classifier(h)

        return out, h

model = GCN()
print(model)














