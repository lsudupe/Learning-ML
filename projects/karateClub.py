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
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)
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

#out, h = model(data.x, data.edge_index)
_, h = model(data.x, data.edge_index) #model() is a instance of the GCN class

print(f"Embedding shape:{list(h.shape)}")
# Detach the tensor from the computation graph and convert it to a NumPy array
h_np = h.detach().cpu().numpy()
plt.scatter(h_np[:, 0], h_np[:, 1], c=data.y, cmap='viridis', alpha=0.7)
plt.show()


###Train the model

#define loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train(data):
    optimizer.zero_grad() #clear gradients
    out, h = model(data.x, data.edge_index) #perform a single forward
    loss = criterion(out[data.train_mask], data.y[data.train_mask]) #compute the loss only with the trainning nodes
    loss.backward() #compute derivates, derive gradients
    optimizer.step() #update parameters

    accuracy = {}
    # Calculate trainning accuracy in our four training mask examples
    predicted_classes = torch.argmax(out[data.train_mask], axis =1)
    target_classes = data.y[data.train_mask]
    accuracy['train'] = torch.mean(
        torch.where(predicted_classes == target_classes, 1, 0).float())

    # Calculate whole graph accuracy
    predicted_classes = torch.argmax(out, axis=1)
    target_classes = data.y
    accuracy['val'] = torch.mean(
        torch.where(predicted_classes == target_classes, 1, 0).float())

    return loss, h, accuracy


### Define trainning loop

for epoch in range(100):
    loss, h, accuracy = train(data)
    h_np = h.detach().cpu().numpy()
    #Visualize the embedding
    if epoch % 10 == 0:
        plt.scatter(h_np[:, 0], h_np[:, 1], c=data.y, cmap='viridis', alpha=0.7)
        plt.show()
        print(f'epoch: {epoch + 1}, loss: {loss}, accuracy: {accuracy}')








