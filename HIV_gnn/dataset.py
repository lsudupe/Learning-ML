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


