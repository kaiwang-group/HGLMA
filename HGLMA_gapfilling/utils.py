import dhg
from dgl.data.utils import load_graphs
import xml.etree.ElementTree as ET
import csv
import json
import os
import pickle as pkl
import numpy as np
import pandas as pd
import torch
import dgl
from dgl import LapPE
from dgl.data.utils import save_graphs
from tqdm import tqdm, trange
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef
from concurrent.futures import as_completed, ProcessPoolExecutor
import networkx as nx
import cobra
import math
from node2vec import Node2Vec
from cobra.util.array import create_stoichiometric_matrix
from cobra.util.solver import linear_reaction_coefficients
import warnings
import re

warnings.filterwarnings("ignore")


def get_data(path, sample):
    """Load a COBRA model and clean it (remove biomass and low-degree reactions)."""
    model = cobra.io.read_sbml_model(os.path.join(path, sample))

    # Handle BiGG models – remove known biomass reactions
    if path.endswith('bigg'):
        biomass_rxns = pd.read_csv('./data/pools/bigg_biomass_reactions.csv')['bigg_id'].to_list()
        rxns = [rxn.id for rxn in model.reactions]
        biomass = [model.reactions.get_by_id(r) for r in rxns if r in biomass_rxns]
    else:
        biomass = list(linear_reaction_coefficients(model).keys())

    model.remove_reactions(biomass, remove_orphans=True)

    # Handle ModelSEED models – remove exchange reactions
    if path.endswith('modelseed'):
        ex_rxns = [r for r in model.reactions if r.id.startswith('EX_')]
        model.remove_reactions(ex_rxns, remove_orphans=True)

    stoichiometric_matrix = create_stoichiometric_matrix(model)
    incidence_matrix = np.abs(stoichiometric_matrix) > 0

    # Remove reactions that involve ≤1 metabolite
    remove_rxn_index = np.sum(incidence_matrix, axis=0) <= 1
    model.remove_reactions([model.reactions[i] for i in np.where(remove_rxn_index)[0]], remove_orphans=True)

    return model, np.abs(create_stoichiometric_matrix(model)) > 0


def process_data(rxn_pool_df, bigg_metabolite, bigg_reaction):
    """Build directed graph, positive/negative hypergraphs, and candidate hypergraph from the universal BiGG pool."""
    glist = load_graphs('mmi_graph.bin')
    g = glist[0][0]

    metabolite = pd.read_csv('metabolite.csv')
    reaction = pd.read_csv('reaction.csv')
    reaction_count = len(reaction)
    metabolite_count = len(metabolite)

    hg_pos = create_hypergraph(pd.read_csv('incidence_matrix_pos.csv'), metabolite_count)
    hg_neg = create_hypergraph(pd.read_csv('incidence_matrix_neg.csv'), metabolite_count)

    print(hg_pos)

    metabolite_new = pd.read_csv('bigg_metabolite.csv')
    metabolite_new_count = len(metabolite_new)

    output_csv = 'incidence_matrix_bigg.csv'

    # Reactions already present in the current model (to be excluded from candidates)
    reaction_df = pd.read_csv('reaction.csv')
    excluded_reactions = set(reaction_df['bigg_id'])
    candidate_reactions = bigg_reaction - excluded_reactions

    metabolite_df = pd.read_csv('bigg_metabolite.csv')
    metabolite_to_node_id = dict(zip(metabolite_df['bigg_id'], metabolite_df['node_id']))
    max_node_id = metabolite_df['node_id'].max()

    if not set(rxn_pool_df.columns).issuperset(candidate_reactions):
        print("Warning: rxn_pool_df columns do not contain all candidate reactions")
    if set(rxn_pool_df.index) != bigg_metabolite:
        print("Warning: rxn_pool_df row index does not exactly match bigg_metabolite")

    reaction_to_hyperedge_id = {rxn: idx for idx, rxn in enumerate(sorted(candidate_reactions))}
    incidence_data = []
    skipped_count = 0

    for rxn in candidate_reactions:
        if rxn not in rxn_pool_df.columns:
            continue
        hyperedge_id = reaction_to_hyperedge_id[rxn]
        rxn_col = rxn_pool_df[rxn]
        for met_id, value in rxn_col.items():
            if value != 0 and met_id in metabolite_to_node_id:
                node_id = metabolite_to_node_id[met_id]
                if node_id <= max_node_id:
                    incidence_data.append({'hyperedge_id': hyperedge_id, 'node_id': node_id})
                else:
                    print(f"Warning: Skipping invalid node_id {node_id} (exceeds max {max_node_id})")
                    skipped_count += 1
            else:
                if value != 0:
                    print(f"Warning: Metabolite {met_id} not found in bigg_metabolite.csv")
                    skipped_count += 1

    incidence_df = pd.DataFrame(incidence_data)
    if incidence_df.empty:
        print("Warning: No hyperedge-node pairs generated – possibly all reactions excluded or no metabolites involved")
    else:
        print(f"Generated {len(incidence_df)} hyperedge-node records, skipped {skipped_count} invalid entries")

    incidence_df.to_csv(output_csv, index=False)
    print(f"Created file {output_csv} with {len(incidence_df)} records")

    hg_bigg = create_hypergraph(pd.read_csv('incidence_matrix_bigg.csv'), metabolite_new_count)

    # Load pre-trained metabolite embeddings
    with open(os.path.join('data/metabolite_emb_2816.pkl'), 'rb') as f:
        embedding_dict = pkl.load(f)

    bigg_ids = metabolite_new['bigg_id']
    embeddings_list = []

    for bigg_id in bigg_ids:
        if bigg_id not in embedding_dict:
            # Try compartment suffix fallback (e.g., _c)
            modified_id = bigg_id.split('_')[0] + '_c'
            if modified_id in embedding_dict:
                embeddings_list.append(embedding_dict[modified_id])
            else:
                # Random embedding for missing metabolites
                embedding = torch.tensor(np.random.rand(2816).astype(np.float32))
                embeddings_list.append(embedding)
        else:
            embeddings_list.append(embedding_dict[bigg_id])

    metabolite_features = torch.stack(embeddings_list)
    return g, hg_pos, hg_neg, hg_bigg, metabolite_features, reaction_count, metabolite_count, metabolite_new_count


def create_hypergraph(df, num_vertices):
    """Convert incidence DataFrame to DHG Hypergraph object."""
    hyperedge_list = []
    for _, group in df.groupby(df.columns[0]):
        hyperedge = list(group[df.columns[1]])
        hyperedge_list.append(hyperedge)
    hg = dhg.Hypergraph(num_vertices, hyperedge_list)
    return hg


def read_xml(xml_content, rxn_pool_df, bigg_metabolite, bigg_reaction):
    """Parse a BiGG SBML string and generate all required CSV files for the current model."""
    root = ET.fromstring(xml_content)
    ns = {
        'sbml': 'http://www.sbml.org/sbml/level3/version1/core',
        'fbc': 'http://www.sbml.org/sbml/level3/version1/fbc/version2'
    }

    # ---------- Extract reactions ----------
    list_of_reactions = root.find('.//sbml:model/sbml:listOfReactions', ns)
    reactions = []
    for reaction in list_of_reactions.findall('sbml:reaction', ns):
        bigg_id = reaction.get('id').replace('R_', '')
        reactions.append(bigg_id)

    # Save reaction → hyperedge_id mapping
    with open('reaction.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['bigg_id', 'hyperedge_id'])
        for idx, bigg_id in enumerate(reactions):
            writer.writerow([bigg_id, idx])

    # ---------- Extract metabolites ----------
    metabolites = set()
    for reaction in list_of_reactions.findall('sbml:reaction', ns):
        for reactant in reaction.findall('sbml:listOfReactants/sbml:speciesReference', ns):
            species_id = reactant.get('species').replace('M_', '')
            metabolites.add(species_id)
        for product in reaction.findall('sbml:listOfProducts/sbml:speciesReference', ns):
            species_id = product.get('species').replace('M_', '')
            metabolites.add(species_id)

    metabolites = sorted(list(metabolites))
    with open('metabolite.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['bigg_id', 'node_id'])
        for idx, bigg_id in enumerate(metabolites):
            writer.writerow([bigg_id, idx])

    # ---------- Extend metabolite list with universal BiGG metabolites ----------
    df = pd.read_csv('metabolite.csv')
    existing_metabolites = set(df['bigg_id'])
    new_metabolites = bigg_metabolite - existing_metabolites

    output_data = df.copy()
    if new_metabolites:
        max_node_id = df['node_id'].max()
        new_rows = [{'bigg_id': met_id, 'node_id': max_node_id + 1 + i}
                    for i, met_id in enumerate(sorted(new_metabolites))]
        output_data = pd.concat([output_data, pd.DataFrame(new_rows)], ignore_index=True)
    else:
        print("No new metabolites to add.")

    output_data.to_csv('bigg_metabolite.csv', index=False)

    # ---------- Build metabolite-metabolite directed edges from universal model ----------
    with open('data/universal_model.json', 'r') as f:
        universal_model = json.load(f)

    metabolite_list = pd.read_csv('bigg_metabolite.csv')['bigg_id'].unique()
    relationships = []
    for rxn in universal_model['reactions']:
        mets = rxn['metabolites']
        for sub_met in metabolite_list:
            if sub_met in mets and mets[sub_met] < 0:  # substrate
                for prod_met, stoich in mets.items():
                    if stoich > 0:
                        relationships.append([sub_met, prod_met])

    pd.DataFrame(relationships, columns=['Substrate', 'Product']).drop_duplicates().to_csv(
        'metabolite_relationships.csv', index=False)

    # Convert to node IDs
    metabolites_df = pd.read_csv('bigg_metabolite.csv')
    rel_df = pd.read_csv('metabolite_relationships.csv')
    met_to_id = dict(zip(metabolites_df['bigg_id'], metabolites_df['node_id']))

    updated_rels = []
    for _, row in rel_df.iterrows():
        if row['Substrate'] != row['Product']:
            s_id = met_to_id.get(row['Substrate'])
            p_id = met_to_id.get(row['Product'])
            if s_id is not None and p_id is not None:
                updated_rels.append({
                    'Substrate': row['Substrate'],
                    'Product': row['Product'],
                    'substrate_id': s_id,
                    'product_id': p_id
                })

    pd.DataFrame(updated_rels).to_csv('mmi.csv', index=False)

    # ---------- Positive incidence matrix (real reactions in the current model) ----------
    reaction_df = pd.read_csv('reaction.csv')
    metabolite_df = pd.read_csv('bigg_metabolite.csv')
    reaction_to_hyperedge = dict(zip(reaction_df['bigg_id'], reaction_df['hyperedge_id']))
    metabolite_to_node = dict(zip(metabolite_df['bigg_id'], metabolite_df['node_id']))

    incidence_data = []

    def parse_sbml_content(content):
        root = ET.fromstring(content)
        ns = {'sbml': 'http://www.sbml.org/sbml/level3/version1/core'}
        for reaction in root.findall('.//sbml:listOfReactions/sbml:reaction', ns):
            rxn_id = reaction.get('id').replace('R_', '')
            hyperedge_id = reaction_to_hyperedge.get(rxn_id)
            if hyperedge_id is None:
                continue

            # Reactants
            reactants = reaction.find('sbml:listOfReactants', ns)
            if reactants is not None:
                for sp in reactants.findall('sbml:speciesReference', ns):
                    met_id = sp.get('species').replace('M_', '')
                    node_id = metabolite_to_node.get(met_id)
                    if node_id is not None:
                        incidence_data.append({'hyperedge_id': hyperedge_id, 'node_id': node_id})

            # Products
            products = reaction.find('sbml:listOfProducts', ns)
            if products is not None:
                for sp in products.findall('sbml:speciesReference', ns):
                    met_id = sp.get('species').replace('M_', '')
                    node_id = metabolite_to_node.get(met_id)
                    if node_id is not None:
                        incidence_data.append({'hyperedge_id': hyperedge_id, 'node_id': node_id})

    parse_sbml_content(xml_content)

    pd.DataFrame(incidence_data).to_csv('incidence_matrix_pos.csv', index=False)

    # ---------- Negative incidence matrix (randomly corrupted hyperedges) ----------
    pos_df = pd.read_csv('incidence_matrix_pos.csv')
    np.random.seed(42)
    neg_data = []
    total_metabolites = len(pd.read_csv('metabolite.csv'))

    for hyperedge_id in pos_df['hyperedge_id'].unique():
        real_nodes = pos_df[pos_df['hyperedge_id'] == hyperedge_id]['node_id'].tolist()
        all_nodes = set(range(total_metabolites))
        possible_nodes = list(all_nodes - set(real_nodes))
        fake_nodes = np.random.choice(possible_nodes, size=len(real_nodes), replace=False)
        for n in fake_nodes:
            neg_data.append({'hyperedge_id': hyperedge_id, 'node_id': n})

    pd.DataFrame(neg_data).to_csv('incidence_matrix_neg.csv', index=False)

    # ---------- Build directed metabolite-metabolite graph with Laplacian eigenvectors ----------
    pos_incidence = pd.read_csv('incidence_matrix_pos.csv')
    os.environ['DGLBACKEND'] = 'pytorch'
    tensor = torch.from_numpy(pos_incidence.values)

    rel_types = [
        ('reaction', 'rmi', 'metabolite'),
        ('metabolite', 'mri', 'reaction')
    ]
    graph_data = {
        rel_types[0]: (tensor[:, 0], tensor[:, 1]),
        rel_types[1]: (tensor[:, 1], tensor[:, 0])
    }
    hetero_graph = dgl.heterograph(graph_data)
    print(hetero_graph)

    mmi = pd.read_csv('mmi.csv')
    src = mmi['substrate_id'].tolist()
    dst = mmi['product_id'].tolist()

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    transform = LapPE(k=3, feat_name='eig')

    g = dgl.graph((src, dst))
    g = g.to(device)
    g = transform(g)
    save_graphs('mmi_graph.bin', g)
    return


def add_padding_idx(vec):
    """Add +1 to all node IDs (reserve 0 for padding in Transformer)."""
    if len(vec.shape) == 1:
        return np.array([np.sort(v + 1).astype('int') for v in tqdm(vec)])
    else:
        return np.sort(vec + 1, axis=-1).astype('int')


def np2tensor_hyper(vec, dtype):
    """Convert numpy hyperedge list to torch tensors."""
    vec = np.asarray(vec)
    if len(vec.shape) == 1:
        return [torch.as_tensor(v, dtype=dtype) for v in vec]
    else:
        return torch.as_tensor(vec, dtype=dtype)


def roc_auc_cuda(y_true, y_pred):
    """CUDA-compatible AUROC & AUPRC calculation."""
    try:
        y_true = y_true.cpu().detach().numpy().reshape((-1, 1))
        y_pred = y_pred.cpu().detach().numpy().reshape((-1, 1))
        return roc_auc_score(y_true, y_pred), average_precision_score(y_true, y_pred)
    except Exception:
        return 0.0, 0.0


def accuracy(output, target):
    """Simple binary accuracy."""
    pred = output >= 0.5
    truth = target >= 0.5
    acc = torch.sum(pred.eq(truth))
    return float(acc) / truth.shape[0]


def pass_(x):
    return x