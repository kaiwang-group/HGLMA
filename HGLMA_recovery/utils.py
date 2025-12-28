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
from tqdm import tqdm


def process_data(rxn_pool_df, bigg_metabolite, bigg_reaction):
    glist = load_graphs('mmi_graph.bin')
    g = glist[0][0]
    metabolite = pd.read_csv('metabolite.csv')
    reaction = pd.read_csv('reaction.csv')
    reaction_count = reaction['bigg_id'].count()
    metabolite_count = len(metabolite)
    hg_pos = create_hypergraph(pd.read_csv('incidence_matrix_pos.csv'), metabolite_count)
    hg_neg = create_hypergraph(pd.read_csv('incidence_matrix_neg.csv'), metabolite_count)

    print(hg_pos)

    metabolite_new = pd.read_csv('bigg_metabolite.csv')
    metabolite_new_count = len(metabolite_new)

    # Check if incidence_matrix_bigg.csv exists
    output_csv = 'incidence_matrix_bigg.csv'
    if os.path.exists(output_csv):
        print(f"File {output_csv} already exists, skipping generation step.")
    else:
        # Read reaction.csv to get reaction IDs to exclude
        reaction_df = pd.read_csv('reaction.csv')
        excluded_reactions = set(reaction_df['bigg_id'])

        # Get valid reactions set (excluding reactions in reaction.csv)
        valid_reactions = bigg_reaction - excluded_reactions

        # Read bigg_metabolite.csv to map metabolite ID to node_id
        metabolite_df = pd.read_csv('bigg_metabolite.csv')
        metabolite_to_node_id = dict(zip(metabolite_df['bigg_id'], metabolite_df['node_id']))

        # Validate rxn_pool_df rows and columns
        if not set(rxn_pool_df.columns).issuperset(valid_reactions):
            print("Warning: rxn_pool_df columns do not contain all valid reactions.")
        if set(rxn_pool_df.index) != bigg_metabolite:
            print("Warning: rxn_pool_df indices do not strictly match bigg_metabolite.")

        # Assign hyperedge_id (starting from 0) for valid reactions
        reaction_to_hyperedge_id = {rxn: idx for idx, rxn in enumerate(sorted(valid_reactions))}

        # Construct hyperedge_id and node_id mapping
        incidence_data = []

        # Iterate through each reaction in rxn_pool_df
        for rxn in valid_reactions:
            if rxn in rxn_pool_df.columns:
                hyperedge_id = reaction_to_hyperedge_id[rxn]
                # Get the column for this reaction
                rxn_col = rxn_pool_df[rxn]
                # Find participating metabolites (non-zero elements)
                for met_id, value in rxn_col.items():
                    if value != 0:  # Metabolite participates in reaction (-1 or 1)
                        if met_id in metabolite_to_node_id:
                            node_id = metabolite_to_node_id[met_id]
                            incidence_data.append({'hyperedge_id': hyperedge_id, 'node_id': node_id})
                        else:
                            print(f"Warning: Metabolite {met_id} not found in bigg_metabolite.csv")

        incidence_df = pd.DataFrame(incidence_data)

        if incidence_df.empty:
            print("Warning: No hyperedge_id, node_id pairs generated. Possibly all reactions excluded or no metabolites involved.")
        else:
            print(f"Generated {len(incidence_df)} hyperedge_id, node_id records.")

        incidence_df.to_csv(output_csv, index=False)
        print(f"Generated file {output_csv} with {len(incidence_df)} records.")

    hg_bigg = create_hypergraph(pd.read_csv('incidence_matrix_bigg.csv'), metabolite_new_count)

    with open(os.path.join('data/metabolite_emb_2816.pkl'), 'rb') as f:
        embedding_dict = pkl.load(f)
    bigg_ids = metabolite_new['bigg_id']
    embeddings_list = []

    for bigg_id in bigg_ids:
        if bigg_id not in embedding_dict:
            modified_bigg_id = bigg_id.split('_')[0] + '_c'
            if modified_bigg_id in embedding_dict:
                bigg_id = modified_bigg_id
            else:
                embedding = torch.tensor(np.random.rand(2816).astype(np.float32))
                embeddings_list.append(embedding)
                continue

        embeddings_list.append(embedding_dict[bigg_id])

    metabolite_features = torch.stack(embeddings_list)
    return g, hg_pos, hg_neg, hg_bigg, metabolite_features, reaction_count, metabolite_count, metabolite_new_count


def create_hypergraph(file, count):
    lists = []
    for _, group in file.groupby(file.columns[0]):
        hyperedge = list(group[group.columns[1]])
        lists.append(hyperedge)
    hg = dhg.Hypergraph(count, lists)
    return hg


def read_xml(xml_content, rxn_pool_df, bigg_metabolite, bigg_reaction):
    root = ET.fromstring(xml_content)
    ns = {
        'sbml': 'http://www.sbml.org/sbml/level3/version1/core',
        'fbc': 'http://www.sbml.org/sbml/level3/version1/fbc/version2'
    }
    list_of_reactions = root.find('.//sbml:model/sbml:listOfReactions', ns)
    reactions = []
    for reaction in list_of_reactions.findall('sbml:reaction', ns):
        bigg_id = reaction.get('id').replace('R_', '')
        reactions.append(bigg_id)

    with open('reaction.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['bigg_id', 'hyperedge_id'])
        for index, bigg_id in enumerate(reactions):
            writer.writerow([bigg_id, index])

    metabolites = set()
    for reaction in list_of_reactions.findall('sbml:reaction', ns):
        # listOfReactants
        for reactant in reaction.findall('sbml:listOfReactants/sbml:speciesReference', ns):
            species_id = reactant.get('species').replace('M_', '')
            metabolites.add(species_id)

        # listOfProducts
        for product in reaction.findall('sbml:listOfProducts/sbml:speciesReference', ns):
            species_id = product.get('species').replace('M_', '')
            metabolites.add(species_id)

    metabolites = sorted(list(metabolites))
    with open('metabolite.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['bigg_id', 'node_id'])  # Write header
        for index, bigg_id in enumerate(metabolites):
            writer.writerow([bigg_id, index])

    # Handle BiGG
    csv_file = 'metabolite.csv'
    df = pd.read_csv(csv_file)
    existing_metabolites = set(df['bigg_id'])
    new_metabolites = bigg_metabolite - existing_metabolites
    output_data = df.copy()

    # Assign node IDs to new metabolites if any
    if not new_metabolites:
        print("No new metabolites to add.")
    else:
        max_node_id = df['node_id'].max()
        new_data = []
        for i, met_id in enumerate(sorted(new_metabolites)):
            new_data.append({'bigg_id': met_id, 'node_id': max_node_id + 1 + i})

        new_df = pd.DataFrame(new_data)
        output_data = pd.concat([output_data, new_df], ignore_index=True)

    output_csv = 'bigg_metabolite.csv'
    output_data.to_csv(output_csv, index=False)
    print(f"Generated new file {output_csv}, containing {len(output_data)} records.")

    with open('data/universal_model.json', 'r') as file:
        model_data = json.load(file)
    metabolites_df = pd.read_csv('bigg_metabolite.csv')
    metabolite_list = metabolites_df['bigg_id'].unique()
    metabolite_relationships = []
    for reaction in model_data['reactions']:
        metabolites_in_reaction = reaction['metabolites']
        for metabolite in metabolite_list:
            if metabolite in metabolites_in_reaction:
                for product, stoichiometry in metabolites_in_reaction.items():
                    if stoichiometry > 0:
                        metabolite_relationships.append([metabolite, product])

    relationship_df = pd.DataFrame(metabolite_relationships, columns=['Metabolite', 'Product'])
    relationship_df = relationship_df.drop_duplicates()
    relationship_df.to_csv('metabolite_relationships.csv', index=False)
    metabolites_df = pd.read_csv('bigg_metabolite.csv')
    relationships_df = pd.read_csv('metabolite_relationships.csv')
    metabolite_to_id = pd.Series(metabolites_df.node_id.values, index=metabolites_df.bigg_id).to_dict()
    updated_relationships = []
    for _, row in relationships_df.iterrows():
        if row['Metabolite'] != row['Product']:
            substrate_id = metabolite_to_id.get(row['Metabolite'])
            product_id = metabolite_to_id.get(row['Product'])
            if substrate_id is not None and product_id is not None:
                updated_relationships.append({
                    'Metabolite': row['Metabolite'],
                    'Product': row['Product'],
                    'substrate_id': substrate_id,
                    'product_id': product_id
                })

    updated_relationships_df = pd.DataFrame(updated_relationships)
    updated_relationships_df.to_csv('mmi.csv', index=False)
    reaction_df = pd.read_csv('reaction.csv')
    metabolite_df = pd.read_csv('bigg_metabolite.csv')
    reaction_to_id = pd.Series(reaction_df.hyperedge_id.values, index=reaction_df.bigg_id).to_dict()
    metabolite_to_id = pd.Series(metabolite_df.node_id.values, index=metabolite_df.bigg_id).to_dict()
    incidence_matrix_data = []

    def parse_xml(file_path):
        root = ET.fromstring(file_path)
        ns = {'sbml': 'http://www.sbml.org/sbml/level3/version1/core'}
        for reaction in root.findall('.//sbml:listOfReactions/sbml:reaction', ns):
            reaction_id = reaction.attrib['id'].replace('R_', '')
            hyperedge_id = reaction_to_id.get(reaction_id)
            if hyperedge_id is not None:
                reactants = reaction.find('sbml:listOfReactants', ns)
                products = reaction.find('sbml:listOfProducts', ns)
                if reactants is not None:
                    for reactant in reactants.findall('sbml:speciesReference', ns):
                        metabolite_id = reactant.attrib['species'].replace('M_', '')
                        node_id = metabolite_to_id.get(metabolite_id)
                        if node_id is not None:
                            incidence_matrix_data.append({
                                'hyperedge_id': hyperedge_id,
                                'node_id': node_id
                            })
                if products is not None:
                    for product in products.findall('sbml:speciesReference', ns):
                        metabolite_id = product.attrib['species'].replace('M_', '')
                        node_id = metabolite_to_id.get(metabolite_id)
                        if node_id is not None:
                            incidence_matrix_data.append({
                                'hyperedge_id': hyperedge_id,
                                'node_id': node_id
                            })

    parse_xml(xml_content)

    incidence_matrix_df = pd.DataFrame(incidence_matrix_data)
    incidence_matrix_df.to_csv('incidence_matrix_pos.csv', index=False)
    df = pd.read_csv('incidence_matrix_pos.csv')
    np.random.seed(42)
    neg_data = []
    hyperedge_ids = df['hyperedge_id'].unique()

    total_metabolites = len(metabolites)
    for hyperedge_id in hyperedge_ids:
        metabolites = df[df['hyperedge_id'] == hyperedge_id]['node_id'].tolist()
        all_possible_metabolites = set(range(0, total_metabolites))
        existing_metabolites = set(metabolites)
        possible_replacements = list(all_possible_metabolites - existing_metabolites)
        random_replacements = np.random.choice(possible_replacements, len(metabolites), replace=False)
        for metabolite in random_replacements:
            neg_data.append({'hyperedge_id': hyperedge_id, 'node_id': metabolite})

    neg_df = pd.DataFrame(neg_data)
    neg_df.to_csv('incidence_matrix_neg.csv', index=False)

    reaction_metabolity = pd.read_csv('incidence_matrix_pos.csv')
    os.environ['DGLBACKEND'] = 'pytorch'
    reaction_metabolity_tensor = torch.from_numpy(reaction_metabolity.values)
    rel_list = [
        ('reaction', 'rmi', 'metabolite'),
        ('metabolite', 'mri', 'reaction')]
    graph_data = {
        rel_list[0]: (reaction_metabolity_tensor[:, 0], reaction_metabolity_tensor[:, 1]),
        rel_list[1]: (reaction_metabolity_tensor[:, 1], reaction_metabolity_tensor[:, 0])
    }
    hetero_graph = dgl.heterograph(graph_data)
    print(hetero_graph)
    mmi = pd.read_csv('mmi.csv')
    substrate_ids = mmi['substrate_id'].tolist()
    product_ids = mmi['product_id'].tolist()
    device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device('cpu')
    transform = LapPE(k=3, feat_name='eig')

    g = dgl.graph((substrate_ids, product_ids))
    g = g.to(device)
    g = transform(g)
    save_graphs('mmi_graph.bin', g)
    return

def add_padding_idx(vec):
    if len(vec.shape) == 1:
        return np.asarray([np.sort(np.asarray(v) + 1).astype('int')
                         for v in tqdm(vec)])
    else:
        vec = np.asarray(vec) + 1
        vec = np.sort(vec, axis=-1)
        return vec.astype('int')

def np2tensor_hyper(vec, dtype):
    vec = np.asarray(vec)
    if len(vec.shape) == 1:
        return [torch.as_tensor(v, dtype=dtype) for v in vec]
    else:
        return torch.as_tensor(vec, dtype = dtype)

def roc_auc_cuda(y_true, y_pred):
    try:
        y_true = y_true.cpu().detach().numpy().reshape((-1, 1))
        y_pred = y_pred.cpu().detach().numpy().reshape((-1, 1))
        return roc_auc_score(
            y_true, y_pred), average_precision_score(
            y_true, y_pred)
    except BaseException:
        return 0.0, 0.0


def accuracy(output, target):
    pred = output >= 0.5
    truth = target >= 0.5
    acc = torch.sum(pred.eq(truth))
    acc = float(acc) * 1.0 / (truth.shape[0] * 1.0)
    return acc


def pass_(x):
    return x