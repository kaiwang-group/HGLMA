import pandas as pd
import torch
import numpy as np
import pickle
from transformers import AutoTokenizer, AutoModel
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data

input_file = 'metabolites.csv'
output_file = 'metabolite_features.pkl'

chemberta_model_name = "seyonec/ChemBERTa-zinc-base-v1"
# Pre-trained model link: https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1
tokenizer = AutoTokenizer.from_pretrained(chemberta_model_name)
chemberta_model = AutoModel.from_pretrained(chemberta_model_name)
chemberta_model.eval()


class GraphMVP(torch.nn.Module):
    def __init__(self, output_dim=2048):
        super(GraphMVP, self).__init__()
        self.gnn = torch.nn.Linear(300, output_dim)

    def forward(self, data):
        return self.gnn(data.x.float().mean(dim=0).unsqueeze(0))


graphmvp_model = GraphMVP(output_dim=2048)
# Pre-trained model link: https://github.com/chao1224/GraphMVP
# graphmvp_model.load_state_dict(torch.load('GraphMVP_pretrained.pth'))
graphmvp_model.eval()


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    AllChem.Compute2DCoords(mol)
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()

    node_f = [atom.GetAtomicNum() for atom in atoms]
    node_features = torch.tensor(node_f, dtype=torch.float).unsqueeze(1)
    # Padding to generic input size for the dummy model, real implementation needs actual atom features
    if node_features.shape[1] < 300:
        padding = torch.zeros(node_features.shape[0], 300 - node_features.shape[1])
        node_features = torch.cat([node_features, padding], dim=1)

    edge_indices = []
    for bond in bonds:
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.append([i, j])
        edge_indices.append([j, i])

    if len(edge_indices) > 0:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return Data(x=node_features, edge_index=edge_index)


df = pd.read_csv(input_file, header=None)
results = {}

with torch.no_grad():
    for index, row in df.iterrows():
        name = row[0]
        smiles = str(row[1]).strip()

        if not smiles or smiles == 'nan':
            feature_vector = torch.randn(2816)
        else:
            inputs = tokenizer(smiles, return_tensors="pt", padding=True, truncation=True, max_length=512)
            chemberta_out = chemberta_model(**inputs)
            chemberta_feat = chemberta_out.last_hidden_state.mean(dim=1).squeeze()

            graph_data = smiles_to_graph(smiles)
            if graph_data is None:
                feature_vector = torch.randn(2816)
            else:
                graphmvp_feat = graphmvp_model(graph_data).squeeze()

                if chemberta_feat.shape[0] != 768:
                    chemberta_feat = torch.randn(768)
                if graphmvp_feat.shape[0] != 2048:
                    graphmvp_feat = torch.randn(2048)

                feature_vector = torch.cat((chemberta_feat, graphmvp_feat), dim=0)

        results[name] = feature_vector.numpy()

with open(output_file, 'wb') as f:
    pickle.dump(results, f)