import matplotlib.pyplot as plt
import pandas as pd

# Load the data
DATA_PATH = "./HIV_gnn/data/HIV.csv"
data = pd.read_csv(DATA_PATH)
data.head()

# Check the data
print(data.shape)
print(data["HIV_active"].value_counts())
# the values are not balance, we need to consider this latter

# Visualize molecules
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt

samples_smiles = data["smiles"][4:30].values
sample_moles = [Chem.MolFromSmiles(smiles) for smiles in samples_smiles]
grid = Draw.MolsToGridImage(sample_moles,
                            molsPerRow =4,
                            subImgSize =(200,200))
plt.imshow(grid)


# check our packages versions
import torch
import torch_geometric
from torch_geometric.data import Data
from tqdm import tqdm

print(f"torch version/t/t: {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
print(f"torch geometric version: {torch_geometric.__version__}")

class MoleculeDataset(Dataset):
    def __init__(self, root, filename, test=False, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        """

        super(MoleculeDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        ''''''
        return HIV.csv

    @property
    def processed_file_names(self):

    def dowloaded(self):
        pass

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0])
        for index, mol in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            mol_obj = Chem.MolFromSmiles(mol["smiles"])
            # get node features
            node_feats = self.get_node_features(mol_obj)
            # get edge features
            edge_feats = self.get_edge_features(mol_obj)
            # get adjacency info
            edge_index = self.get_adjacency_info(mol_obj)
            # get labels info
            label = self.get_labels()

            # Create data object
            data = Data(x = node_feats,
                        edge_index = edge_index,
                        edge_attr = edge_feats,
                        y= label,
                        smiles = mol["smiles"]
                        )

    def get_node_features(self):
    '''This will return a matrix /2d array of shape
    [number of nodes, node feature size]'''
        all_node_features = []
        for atom in mol.GetAtoms():
            node_feats = []
            # Feature 1: Atomic number
            node_feats.append(atom.GetAtomicNum())
            # Feature 2: Atom degree
            node_feats.append(atom.GetDegree())
            # Feature 3: Formal charge
            node_feats.append(atom.GetFormalCharge())
            # Feature 4: Hybridization
            node_feats.append(atom.GetHybridization())
            # Feature 5: Aromaticity
            node_feats.append(atom.GetIsAromatic())

        # append node features to matrix
        all_node_features.append(node_feats)

        all_node_features = np.sarray(all_node_features)
        return torch.tensor(all_node_features, dtype=torch.float)

    def get_edge_features(self, mol):
        all_bond_features = []
        for edge in mol.GetBonds():
            edge_feats = []
            edge_feats.append(e)

    def get_adjacency_info(self, mol):

    def get_labels(self):


for index, mol in tqdm(data.iterrows(), total=data.shape[0]):
    a = index
    b = mol

type(mol_obj)
node_feats = get_node_features(mol_obj)

molecule_class = rdkit.Chem.rdchem.Mol
molecule_class
print(dir(molecule_class))

a =data.iterrows()
a.head()

mol["HIV_active"]




