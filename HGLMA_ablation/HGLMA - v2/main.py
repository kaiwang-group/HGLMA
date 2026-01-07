import logging
import torch
import os
import csv
import pickle as pkl
import numpy as np
from dhg.random import set_seed
import pandas as pd
from tqdm import tqdm
import argparse
from sklearn.model_selection import KFold
import glob
from torch.nn.utils.rnn import pad_sequence
from scipy.sparse import csr_matrix
from scipy.sparse import vstack as s_vstack
import torch.nn.functional as F
import time
import warnings
from model import *
from utils import *
from train import *
import matplotlib as mpl



def setup_logging(args):
    log_dir = os.path.join(args.save_path, 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.info(f"Log file will be saved to: {os.path.abspath(log_file)}")
    return logger


def parse_args():
    parser = argparse.ArgumentParser(description='train Multi-HGNN for missing reaction prediction.')
    parser.add_argument('--external_epochs', default=1, type=int, help='maximum training external_epochs')
    parser.add_argument('--internal_epochs', default=300, type=int, help='maximum training epochs per fold')
    parser.add_argument('--lr', default=0.0005, type=float, help='learning rate')
    parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--in_dim', default=512, type=int, help='dim of input embedding')
    parser.add_argument('--h_dim', default=256, type=str, help='dim of hidden embedding')
    parser.add_argument('--out_dim', default=64, type=int, help='dim of output embedding')
    parser.add_argument('--cuda', default=1, type=int, help='gpu index')
    parser.add_argument('--k_fold', default=5, type=int, help='k-fold cross validation')
    parser.add_argument('--k', default=2, type=int, help='k')
    parser.add_argument('--num_dgnn', default=2, type=int,
                        help='the number of layers in the directed graph neural network')
    parser.add_argument('--num_hgnn', default=1, type=int, help='the number of layers in the hypergraph neural network')
    parser.add_argument('--data', type=str, default='model')
    parser.add_argument('--TRY', action='store_true')
    parser.add_argument('--FILTER', action='store_true')
    parser.add_argument('--grid', type=str, default='')
    parser.add_argument('--remark', type=str, default='')
    parser.add_argument('--dimensions', type=int, default=64,
                        help='Number of dimensions. Default is 64.')
    parser.add_argument('--rw', type=float, default=0.01,
                        help='The weight of reconstruction of adjacency matrix loss. Default is ')
    parser.add_argument('-d', '--diag', type=str, default='True',
                        help='Use the diag mask or not')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device('cpu')
    args = parse_args()

    batch_size = 96
    bottle_neck = args.dimensions

    metric_collector = {
        'valid_bce_loss': [],
        'valid_acc': [],
        'valid_auroc': [],
        'valid_auprc': [],
        'valid_precision': [],
        'valid_recall': [],
        'valid_f1': [],
        'valid_mcc': []
    }

    xml_files = glob.glob(os.path.join('BiGG Models', '*.xml'))
    for xml_file in xml_files:
        model_name = os.path.splitext(os.path.basename(xml_file))[0]
        args.save_path = os.path.join('./checkpoints/', args.data, model_name)
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        logger = setup_logging(args)
        logger.info(f'Current model ID: {model_name}')

        with open(xml_file, 'r', encoding='utf-8') as file:
            xml_content = file.read()
        read_xml(xml_content)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path_pos = os.path.join(current_dir, 'incidence_matrix_pos.csv')
        file_path_neg = os.path.join(current_dir, 'incidence_matrix_neg.csv')

        df_pos = pd.read_csv(file_path_pos)
        df_neg = pd.read_csv(file_path_neg)

        edges_pos = df_pos.groupby('hyperedge_id')['node_id'].apply(list).to_list()
        edges_neg = df_neg.groupby('hyperedge_id')['node_id'].apply(list).to_list()

        edges_weight_pos = np.ones(len(edges_pos), dtype='float32')
        edges_weight_neg = np.ones(len(edges_neg), dtype='float32')
        reaction = pd.read_csv('reaction.csv')
        reaction_to_hyperedge = dict(zip(reaction.index, reaction['bigg_id']))

        fold_assignments = []

        # Get features using Node2Vec
        g, hg_pos, hg_neg, initial_features, reaction_count, metabolite_count = process_data()

        kf = KFold(n_splits=args.k_fold, shuffle=True, random_state=0)
        m_emb = initial_features
        results = []


        for metric in metric_collector.values():
            metric.clear()

        for fold, (train_set, valid_set) in enumerate(kf.split(np.arange(reaction_count))):
            for idx in valid_set:
                bigg_id = reaction.iloc[idx]['bigg_id']
                fold_assignments.append({'bigg_id': bigg_id, 'fold': fold})

            train_hyperedge_ids_set = [reaction_to_hyperedge[i] for i in train_set]
            valid_hyperedge_ids_set = [reaction_to_hyperedge[i] for i in valid_set]

            train_edges_pos_set = [edges_pos[i] for i in train_hyperedge_ids_set]
            valid_edges_pos_set = [edges_pos[i] for i in valid_hyperedge_ids_set]
            train_edges_neg_set = [edges_neg[i] for i in train_hyperedge_ids_set]
            valid_edges_neg_set = [edges_neg[i] for i in valid_hyperedge_ids_set]

            train_weight_pos_set = edges_weight_pos[train_hyperedge_ids_set]
            valid_weight_pos_set = edges_weight_pos[valid_hyperedge_ids_set]
            train_weight_neg_set = edges_weight_neg[train_hyperedge_ids_set]
            valid_weight_neg_set = edges_weight_neg[valid_hyperedge_ids_set]

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
            optimizer = torch.optim.Adam(params_list, lr=args.learning_rate, weight_decay=args.wd)
            m_emb = m_emb.to(device)
            g = g.to(device)
            net = net.to(device)
            classifier_model = classifier_model.to(device)
            train_set = torch.tensor(train_set).long()
            valid_set = torch.tensor(valid_set).long()

            loss_f = F.binary_cross_entropy
            epoch_loss = []
            start_time = time.time()

            for epoch in tqdm(range(args.external_epochs), desc=f"Training Fold {fold}"):
                net, classifier_model, fold_best_metrics = train(args, net, classifier_model, m_emb, g, hg_pos, hg_neg,
                                                                 train_set, valid_set, metabolite_count, reaction_count,
                                                                 loss_f,
                                                                 training_data=(
                                                                 train_edges_pos_set, train_weight_pos_set,
                                                                 train_edges_neg_set, train_weight_neg_set),
                                                                 validation_data=(
                                                                 valid_edges_pos_set, valid_weight_pos_set,
                                                                 valid_edges_neg_set, valid_weight_neg_set),
                                                                 optimizer=[optimizer], epochs=args.internal_epochs,
                                                                 batch_size=batch_size, fold=fold,
                                                                 model_name=model_name)

                for metric_name in metric_collector.keys():
                    metric_collector[metric_name].append(fold_best_metrics[metric_name])

                all_folds_best_metrics.append(fold_best_metrics)

                if len(metric_collector['valid_acc']) > 4:
                    logger.info("\n[Cross-Validation Results ± Std]")
                    logger.info(
                        f"  Loss: {np.mean(metric_collector['valid_bce_loss']):.4f} ")
                    logger.info(
                        f"  ACC: {100 * np.mean(metric_collector['valid_acc']):.1f}%")
                    logger.info(
                        f"  Precision: {np.mean(metric_collector['valid_precision']):.3f}")
                    logger.info(
                        f"  Recall: {np.mean(metric_collector['valid_recall']):.3f}")
                    logger.info(
                        f"  F1: {np.mean(metric_collector['valid_f1']):.3f}")
                    logger.info(
                        f"  MCC: {np.mean(metric_collector['valid_mcc']):.3f}")
                    logger.info(
                        f"  AUROC: {np.mean(metric_collector['valid_auroc']):.3f}")
                    logger.info(
                        f"  AUPRC: {np.mean(metric_collector['valid_auprc']):.3f}")

        fold_assignments_file = os.path.join(args.save_path, 'fold_assignments.csv')
        with open(fold_assignments_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['bigg_id', 'fold'])
            writer.writeheader()
            for assignment in fold_assignments:
                writer.writerow(assignment)
        logger.info(f'Fold assignments saved to: {fold_assignments_file}')