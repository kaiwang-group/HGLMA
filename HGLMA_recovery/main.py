import logging
import torch
import os
import pickle as pkl
import numpy as np
import cobra
from dhg.random import set_seed
import pandas as pd
from tqdm import tqdm
import argparse
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import glob
import torch.nn.functional as F
import time
import random
import warnings
from model import *
from utils import *
from train import *
import matplotlib as mpl
import multiprocessing

mpl.use("Agg")
cpu_num = multiprocessing.cpu_count()

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

warnings.filterwarnings("ignore")


def setup_logging(args):
    log_dir = os.path.join(args.save_path, 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, f'GCF_845.2_ALL_TOP{args.top}_recovery_SMILESlr{args.lr}_training_log_{time.strftime("%Y%m%d_%H%M%S")}.txt')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Log file will be saved to: {os.path.abspath(log_file)}")
    return logger


def parse_args():
    parser = argparse.ArgumentParser(description='train Multi-HGNN for missing reaction prediction.')
    parser.add_argument('--top', default=100, type=int, help='Number of top score ranks to select')
    parser.add_argument('--external_epochs', default=1, type=int, help='Maximum training external epochs (multi: 100, sagnn: 1)')
    parser.add_argument('--internal_epochs', default=400, type=int, help='Maximum training epochs per fold (sagnn: 300)')
    parser.add_argument('--lr', default=0.0005, type=float, help='Learning rate (sagnn: 0.003, multi: 0.002)')
    parser.add_argument('--wd', default=5e-4, type=float, help='Weight decay')
    parser.add_argument('--in_dim', default=512, type=int, help='Dim of input embedding')
    parser.add_argument('--h_dim', default=256, type=int, help='Dim of hidden embedding')
    parser.add_argument('--out_dim', default=64, type=int, help='Dim of output embedding')
    parser.add_argument('--cuda', default=1, type=int, help='GPU index')
    parser.add_argument('--k_fold', default=5, type=int, help='K-fold cross validation')
    parser.add_argument('--k', default=2, type=int, help='k')
    parser.add_argument('--num_dgnn', default=2, type=int, help='The number of layers in the directed graph neural network')
    parser.add_argument('--num_hgnn', default=1, type=int, help='The number of layers in the hypergraph neural network')
    parser.add_argument('--data', type=str, default='ramani')
    parser.add_argument('--TRY', action='store_true')
    parser.add_argument('--FILTER', action='store_true')
    parser.add_argument('--grid', type=str, default='')
    parser.add_argument('--remark', type=str, default='')
    parser.add_argument('--dimensions', type=int, default=64, help='Number of dimensions. Default is 64.')
    parser.add_argument('--rw', type=float, default=0.01, help='The weight of reconstruction of adjacency matrix loss.')
    parser.add_argument('-d', '--diag', type=str, default='True', help='Use the diag mask or not')

    args = parser.parse_args()
    args.model_name = 'model_{}_'.format(args.data)
    args.model_name += args.remark

    args.save_path = os.path.join('./checkpoints/', args.data, args.model_name)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    return args


if __name__ == "__main__":
    set_seed(0)
    device = torch.device(f"cuda:{args.cuda}") if torch.cuda.is_available() else torch.device('cpu')
    args = parse_args()
    logger = setup_logging(args)
    batch_size = 96
    bottle_neck = args.dimensions
    train_type = 'hyper'

    universe_pool = cobra.io.read_sbml_model('./data/pools/bigg_universe.xml')
    universe_pool_copy = universe_pool.copy()
    rxn_pool_df = cobra.util.array.create_stoichiometric_matrix(
        universe_pool_copy, array_type='DataFrame'
    )
    bigg_metabolite = set(rxn_pool_df.index)
    bigg_reaction = set(rxn_pool_df.columns)

    xml_files = glob.glob(os.path.join('BiGG Models', '*.xml'))
    # Loop through each BiGG model
    for xml_file in xml_files:
        model_name = os.path.splitext(os.path.basename(xml_file))[0]
        logger.info(f'Current model ID: {model_name}')
        with open(xml_file, 'r', encoding='utf-8') as file:
            xml_content = file.read()
        read_xml(xml_content, rxn_pool_df, bigg_metabolite, bigg_reaction)

        g, hg_pos, hg_neg, hg_bigg, initial_features, reaction_count, metabolite_count, metabolite_new_count = process_data(rxn_pool_df, bigg_metabolite, bigg_reaction)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path_pos = os.path.join(current_dir, 'incidence_matrix_pos.csv')
        file_path_neg = os.path.join(current_dir, 'incidence_matrix_neg.csv')
        file_path_bigg = os.path.join(current_dir, 'incidence_matrix_bigg.csv')

        # Read CSV
        df_pos = pd.read_csv(file_path_pos)
        df_neg = pd.read_csv(file_path_neg)
        df_bigg = pd.read_csv(file_path_bigg)

        # Group to get hyperedges
        edges_pos = df_pos.groupby('hyperedge_id')['node_id'].apply(list).to_list()
        edges_neg = df_neg.groupby('hyperedge_id')['node_id'].apply(list).to_list()
        edges_bigg = df_bigg.groupby('hyperedge_id')['node_id'].apply(list).to_list()

        # Initialize weights
        edges_weight_pos = np.ones(len(edges_pos), dtype='float32')
        edges_weight_neg = np.ones(len(edges_neg), dtype='float32')
        edges_weight_bigg = np.ones(len(edges_bigg), dtype='float32')

        # Read reaction.csv, map reaction index to hyperedge_id
        reaction = pd.read_csv('reaction.csv')
        reaction_to_hyperedge = dict(zip(reaction.index, reaction['hyperedge_id']))

        kf = KFold(n_splits=args.k_fold, shuffle=True, random_state=0)
        m_emb = initial_features.to(device)
        all_folds_best_metrics = []  # Store best metrics for each fold

        for fold, (train_set, valid_set) in enumerate(kf.split(np.arange(reaction_count))):
            logger.info(f'Fold {fold + 1}/{args.k_fold}')

            train_hyperedge_ids_set = [reaction_to_hyperedge[i] for i in train_set]
            valid_hyperedge_ids_set = [reaction_to_hyperedge[i] for i in valid_set]

            train_edges_pos_set = [edges_pos[i] for i in train_hyperedge_ids_set]
            valid_edges_pos_set = [edges_pos[i] for i in valid_hyperedge_ids_set]
            train_edges_neg_set = [edges_neg[i] for i in train_hyperedge_ids_set]
            valid_edges_neg_set = edges_bigg

            train_weight_pos_set = edges_weight_pos[train_hyperedge_ids_set]
            valid_weight_pos_set = edges_weight_pos[valid_hyperedge_ids_set]
            train_weight_neg_set = edges_weight_neg[train_hyperedge_ids_set]
            valid_weight_neg_set = edges_weight_bigg

            # Network Architecture
            net = Model(args.in_dim, args.h_dim, args.out_dim, args.num_dgnn, args.num_hgnn, use_bn=True)
            classifier_model = Classifier(
                n_head=8,
                d_model=args.dimensions,
                d_k=16,
                d_v=16,
                node_embedding=None,  # Placeholder, node embedding is dynamically called
                metabolite_count=metabolite_count,
                diag_mask=args.diag,
                bottle_neck=bottle_neck).to(device)

            # Optimizer collects parameters from both net and classifier
            params_list = list(set(list(classifier_model.parameters()) + list(net.parameters())))
            optimizer = torch.optim.Adam(params_list, lr=args.lr, weight_decay=args.wd)

            g = g.to(device)
            net = net.to(device)
            classifier_model = classifier_model.to(device)
            train_set = torch.tensor(train_set).long()
            valid_set = torch.tensor(valid_set).long()

            loss_f = F.binary_cross_entropy
            epoch_loss = []
            start_time = time.time()

            for epoch in tqdm(range(args.external_epochs), desc=f"Training Fold {fold + 1}"):
                net, classifier_model, best_metrics = train(
                    args, net, classifier_model, m_emb, g, hg_pos, hg_neg, hg_bigg, train_set, valid_set,
                    metabolite_count, reaction_count, loss_f,
                    training_data=(train_edges_pos_set, train_weight_pos_set, train_edges_neg_set, train_weight_neg_set),
                    validation_data=(valid_edges_pos_set, valid_weight_pos_set, valid_edges_neg_set, valid_weight_neg_set),
                    optimizer=[optimizer],
                    epochs=args.internal_epochs,
                    batch_size=batch_size
                )

                # Save model for the current fold
                checkpoint = {
                    'net': net.state_dict(),
                    'classifier_model': classifier_model.state_dict(),
                    'epoch': best_metrics['epoch'],
                    'metrics': best_metrics
                }
                fold_save_path = os.path.join(args.save_path, f'fold_{fold + 1}_best_model.pth')
                torch.save(checkpoint, fold_save_path)
                logger.info(f'Saved best model for fold {fold + 1} to {fold_save_path}')

            # Store best metrics for the current fold
            all_folds_best_metrics.append(best_metrics)
            logger.info(f'Fold {fold + 1} Best Metrics: Epoch {best_metrics["epoch"]}, '
                        f'Loss: {best_metrics["valid_bce_loss"]:.4f}, '
                        f'Top-{args.top} Positive Ratio: {best_metrics["valid_positive_ratio"]:.4f}')

        # Calculate and log average metrics across all folds
        valid_bce_losses = [m['valid_bce_loss'] for m in all_folds_best_metrics]
        valid_positive_ratios = [m['valid_positive_ratio'] for m in all_folds_best_metrics]

        logger.info('\n=== Cross-Validation Summary ===')
        for fold, metrics in enumerate(all_folds_best_metrics, 1):
            logger.info(f'Fold {fold}: Epoch {metrics["epoch"]}, '
                        f'Loss: {metrics["valid_bce_loss"]:.4f}, '
                        f'Top-{args.top} Positive Ratio: {metrics["valid_positive_ratio"]:.4f}')

        mean_bce_loss = np.mean(valid_bce_losses)
        std_bce_loss = np.std(valid_bce_losses)
        mean_positive_ratio = np.mean([r.cpu().item() if torch.is_tensor(r) else r for r in valid_positive_ratios])
        std_positive_ratio = np.std([r.cpu().item() if torch.is_tensor(r) else r for r in valid_positive_ratios])

        logger.info(f'\nAverage Metrics Across {args.k_fold} Folds:')
        logger.info(f'  BCE Loss: {mean_bce_loss:.4f} ± {std_bce_loss:.4f}')
        logger.info(f'  Top-{args.top} Positive Ratio: {mean_positive_ratio:.4f} ± {std_positive_ratio:.4f}')