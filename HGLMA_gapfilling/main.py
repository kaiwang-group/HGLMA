import torch
import os
import csv
import pickle as pkl
import numpy as np
import cobra
from dhg.random import set_seed
import pandas as pd
from tqdm import tqdm
import argparse
import glob
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import warnings
from model import *
from utils import *
from train import *
from similarity import *
from phenotypes import *
import matplotlib as mpl

mpl.use("Agg")
import multiprocessing

cpu_num = multiprocessing.cpu_count()

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description='train Multi-HGNN for missing reaction prediction.')
    parser.add_argument('--run', default=3, type=int, help='Number of prediction runs')
    parser.add_argument('--external_epochs', default=1, type=int,
                        help='Maximum external training epochs')
    parser.add_argument('--internal_epochs', default=100, type=int,
                        help='Maximum training epochs per fold')
    parser.add_argument('--lr', default=0.0005, type=float, help='Learning rate')
    parser.add_argument('--wd', default=5e-4, type=float, help='Weight decay')
    parser.add_argument('--in_dim', default=512, type=int, help='Dimension of input embedding')
    parser.add_argument('--h_dim', default=256, type=int, help='Dimension of hidden embedding')
    parser.add_argument('--out_dim', default=64, type=int, help='Dimension of output embedding')
    parser.add_argument('--cuda', default=1, type=int, help='GPU index')
    parser.add_argument('--k_fold', default=5, type=int, help='K-fold cross validation')
    parser.add_argument('--k', default=2, type=int, help='k')
    parser.add_argument('--num_dgnn', default=2, type=int, help='Number of layers in the directed graph neural network')
    parser.add_argument('--num_hgnn', default=1, type=int, help='Number of layers in the hypergraph neural network')
    parser.add_argument('--data', type=str, default='ramani')
    parser.add_argument('--TRY', action='store_true')
    parser.add_argument('--FILTER', action='store_true')
    parser.add_argument('--grid', type=str, default='')
    parser.add_argument('--remark', type=str, default='')
    parser.add_argument('--dimensions', type=int, default=64,
                        help='Number of dimensions. Default is 64.')
    parser.add_argument('--rw', type=float, default=0.01,
                        help='Weight of adjacency matrix reconstruction loss. Default is 0.01')
    parser.add_argument('-d', '--diag', type=str, default='True',
                        help='Use diagonal mask or not')

    args = parser.parse_args()
    args.model_name = 'model_{}_'.format(args.data)  # Fixed model name format
    args.model_name += args.remark  # Append user remark

    args.save_path = os.path.join(
        './checkpoints/', args.data, args.model_name)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    return args


if __name__ == "__main__":
    set_seed(0)
    device = torch.device("cuda:1" if torch.cuda.is_available() else torch.device('cpu'))
    args = parse_args()
    # Monkey-patch optlang Container to skip duplicate reaction checks

    batch_size = 96
    bottle_neck = args.dimensions
    train_type = 'hyper'

    # Create result directories
    os.system("mkdir results")
    os.system("mkdir results/predicted_scores")
    os.system("mkdir results/similarity_scores")
    os.system("mkdir results/gaps")

    # Load universal BiGG reaction pool
    universe_pool = cobra.io.read_sbml_model('./data/pools/bigg_universe.xml')
    universe_pool_copy = universe_pool.copy()

    # Detach solver to avoid conflicts during matrix creation
    universe_pool_copy._solver = None

    # Build stoichiometric matrix for the entire BiGG universe
    rxn_pool_df = cobra.util.array.create_stoichiometric_matrix(
        universe_pool_copy, array_type='DataFrame'
    )
    bigg_metabolite = set(rxn_pool_df.index)
    bigg_reaction = set(rxn_pool_df.columns)

    # Process each BiGG genome-scale model
    xml_files = glob.glob(os.path.join('BiGG Models', '*.xml'))
    for xml_file in xml_files:
        model_name = os.path.splitext(os.path.basename(xml_file))[0]
        print(f'Current model ID: {model_name}')  # TODO: replace print with logger

        # Parse the current model's XML content
        with open(xml_file, 'r', encoding='utf-8') as file:
            xml_content = file.read()
        read_xml(xml_content, rxn_pool_df, bigg_metabolite, bigg_reaction)

        # Build graphs and hypergraphs from the stoichiometric matrix
        g, hg_pos, hg_neg, hg_bigg, initial_features, reaction_count, metabolite_count, metabolite_new_count = process_data(
            rxn_pool_df, bigg_metabolite, bigg_reaction)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path_pos = os.path.join(current_dir, 'incidence_matrix_pos.csv')
        file_path_neg = os.path.join(current_dir, 'incidence_matrix_neg.csv')
        file_path_bigg = os.path.join(current_dir, 'incidence_matrix_bigg.csv')

        # Load incidence matrices (hyperedges)
        df_pos = pd.read_csv(file_path_pos)
        df_neg = pd.read_csv(file_path_neg)
        df_bigg = pd.read_csv(file_path_bigg)

        # Group by hyperedge_id to obtain list of nodes per hyperedge
        edges_pos = df_pos.groupby('hyperedge_id')['node_id'].apply(list).to_list()
        edges_neg = df_neg.groupby('hyperedge_id')['node_id'].apply(list).to_list()
        edges_bigg = df_bigg.groupby('hyperedge_id')['node_id'].apply(list).to_list()

        # Initialize uniform hyperedge weights
        edges_weight_pos = np.ones(len(edges_pos), dtype='float32')
        edges_weight_neg = np.ones(len(edges_neg), dtype='float32')
        edges_weight_bigg = np.ones(len(edges_bigg), dtype='float32')

        # Mapping from reaction index to hyperedge_id (used later)
        reaction = pd.read_csv('reaction.csv')
        reaction_to_hyperedge = dict(zip(reaction.index, reaction['hyperedge_id']))

        # Training data (existing positive/negative hyperedges)
        train_edges_pos_set = edges_pos
        train_edges_neg_set = edges_neg
        train_weight_pos_set = edges_weight_pos
        train_weight_neg_set = edges_weight_neg

        # Validation data (candidate reactions from BiGG universe)
        valid_edges_set = edges_bigg
        valid_weight_set = edges_weight_bigg

        # Prepare per-model prediction score file
        results_dir = os.path.join(current_dir, 'results', 'predicted_scores')
        os.makedirs(results_dir, exist_ok=True)
        scores_save_path = os.path.join(results_dir, f'{model_name}.csv')
        if os.path.exists(scores_save_path):
            os.remove(scores_save_path)  # Remove old file to ensure clean overwrite

        # Run multiple independent trainings and append each run's scores as a new column
        for run_idx in range(args.run):
            print(f'Starting run {run_idx + 1}/{args.run} for model: {model_name}')

            # Re-initialize model and optimizer for each run
            net = Model(args.in_dim, args.h_dim, args.out_dim, args.num_dgnn, args.num_hgnn, use_bn=True)
            classifier_model = Classifier(
                n_head=8,
                d_model=args.dimensions,
                d_k=16,
                d_v=16,
                node_embedding=None,
                metabolite_count=metabolite_count,
                diag_mask=args.diag,
                bottle_neck=bottle_neck).to(device)

            params_list = list(set(list(classifier_model.parameters()) + list(net.parameters())))
            optimizer = torch.optim.Adam(params_list, lr=args.lr, weight_decay=args.wd)

            m_emb = initial_features.to(device)
            g = g.to(device)
            net = net.to(device)
            classifier_model = classifier_model.to(device)
            loss_f = F.binary_cross_entropy

            # Training phase
            net, classifier_model, best_metrics = train(
                args, net, classifier_model, m_emb, g, hg_pos, hg_neg, hg_bigg,
                metabolite_count, reaction_count, loss_f,
                training_data=(train_edges_pos_set, train_weight_pos_set,
                               train_edges_neg_set, train_weight_neg_set),
                validation_data=(valid_edges_set, valid_weight_set),
                optimizer=[optimizer],
                epochs=args.internal_epochs,
                batch_size=batch_size)

            # Load the best checkpoint saved during training
            checkpoint = torch.load(os.path.join(args.save_path, 'best_model.pth'))
            net.load_state_dict(checkpoint['net'])
            classifier_model.load_state_dict(checkpoint['classifier_model'])

            # Predict likelihood scores for all candidate reactions
            scores = eval_epoch(args, net, classifier_model, m_emb, g, hg_pos, hg_neg, hg_bigg, loss_f,
                                validation_data=(valid_edges_set, valid_weight_set), batch_size=batch_size)

            # Map hyperedge index back to BiGG reaction ID
            reaction_df = pd.read_csv('reaction.csv')
            excluded_reactions = set(reaction_df['bigg_id'])
            valid_reactions = sorted(bigg_reaction - excluded_reactions)
            hyperedge_to_bigg_id = {idx: rxn for idx, rxn in enumerate(valid_reactions)}

            # Save scores
            scores_np = scores.cpu().numpy()
            scores_data = pd.DataFrame({
                'bigg_id': [hyperedge_to_bigg_id.get(hid, f'unknown_{hid}') for hid in range(len(scores_np))],
                'score': scores_np
            })

            # Append current run's scores as a new column
            if os.path.exists(scores_save_path):
                existing_df = pd.read_csv(scores_save_path)
                score_cols = [col for col in existing_df.columns if col.startswith('score_')]
                new_score_col = f'score_{len(score_cols)}'
                existing_df[new_score_col] = scores_data['score']
                existing_df.to_csv(scores_save_path, index=False)
            else:
                scores_data.rename(columns={'score': 'score_0'}, inplace=True)
                scores_data.to_csv(scores_save_path, index=False)

        # Compute mean similarity between candidate reactions and existing reactions in the model
        get_similarity_score(top_N=2000)

        # Predict metabolic phenotypes after gap-filling
        phenotypes()