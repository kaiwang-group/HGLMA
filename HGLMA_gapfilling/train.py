import logging
from model import *
from utils import *
import torch
import math
import os
import csv
import pickle as pkl
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import time
from sklearn.model_selection import KFold
import glob
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, \
    average_precision_score
from scipy.sparse import csr_matrix
from scipy.sparse import vstack as s_vstack
import warnings

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def train_batch_hyperedge(classifier_model, loss_f, batch_edges_pos, batch_edge_weight_pos, batch_edges_neg,
                          batch_edge_weight_neg, batch_y_pos, batch_y_neg):
    device = next(classifier_model.parameters()).device
    x_pos = torch.tensor(batch_edges_pos, dtype=torch.long, device=device)  # Convert int64 to torch.long
    x_neg = torch.tensor(batch_edges_neg, dtype=torch.long, device=device)
    w_pos = torch.tensor(batch_edge_weight_pos, device=device)  # Weight dtype inferred automatically
    w_neg = torch.tensor(batch_edge_weight_neg, device=device)

    # When no separate labels are provided (standard link prediction setting)
    if len(batch_y_pos) == 0:
        x = torch.cat([x_pos, x_neg])  # Concatenate positive and negative hyperedges
        w = torch.cat([w_pos, w_neg])  # Concatenate corresponding weights
        y = torch.cat([torch.ones(len(x_pos), device=device),
                       torch.zeros(len(x_neg), device=device)])  # Positive=1, Negative=0

        # Randomly shuffle the batch
        index = torch.randperm(len(x), device=device)
        x, y, w = x[index], y[index], w[index]

    # Forward pass through the hyperedge classifier
    pred = classifier_model(x, return_recon=False)
    pred = pred.squeeze(1)

    # Binary cross-entropy with sample weights
    loss = loss_f(pred, y, weight=w)

    return pred, y, loss


def train_epoch(args, net, classifier_model, m_emb, g, hg_pos, hg_neg, hg_bigg, metabolite_count, reaction_count,
                loss_f, training_data, optimizer, batch_size):
    net.train()
    classifier_model.train()

    edges_pos, edge_weight_pos, edges_neg, edge_weight_neg = training_data

    # Pad all hyperedges to the same length (required by Transformer-based classifier)
    max_len = max(len(edge) for edge in edges_pos)
    edges_pos_padded = [edge + [0] * (max_len - len(edge)) for edge in edges_pos]
    edges_neg_padded = [edge + [0] * (max_len - len(edge)) for edge in edges_neg]

    edges_pos = torch.tensor(edges_pos_padded, dtype=torch.long, device=device)
    edges_neg = torch.tensor(edges_neg_padded, dtype=torch.long, device=device)

    # Randomly shuffle training hyperedges
    index = torch.randperm(len(edges_pos)).numpy()
    edges_pos = edges_pos[index]
    edge_weight_pos = edge_weight_pos[index]
    edges_neg = edges_neg[index]
    edge_weight_neg = edge_weight_neg[index]

    bce_total_loss = 0
    acc_list, y_list, pred_list = [], [], []


    y_pos = torch.tensor([])
    y_neg = torch.tensor([])

    batch_num = int(math.floor(len(edges_pos) / batch_size))

    for i in range(batch_num):
        # Zero gradients
        for opt in optimizer:
            opt.zero_grad()

        # Extract current batch
        batch_edges_pos = edges_pos[i * batch_size:(i + 1) * batch_size]
        batch_edge_weight_pos = edge_weight_pos[i * batch_size:(i + 1) * batch_size]
        batch_edges_neg = edges_neg[i * batch_size:(i + 1) * batch_size]
        batch_edge_weight_neg = edge_weight_neg[i * batch_size:(i + 1) * batch_size]


        batch_y_pos = ""
        batch_y_neg = ""


        X1, X2, Y_pos, Y_neg = net(m_emb, g, hg_bigg, hg_bigg)

        if X1.dim() > 2:
            X1 = X1.squeeze()  # Ensure shape is (metabolite_count, bottle_neck)


        classifier_model.set_node_embedding(X1)

        # Downstream hyperedge classifier forward pass
        pred, batch_y, loss = train_batch_hyperedge(
            classifier_model, loss_f, batch_edges_pos, batch_edge_weight_pos,
            batch_edges_neg, batch_edge_weight_neg, batch_y_pos, batch_y_neg)

        # Track accuracy per batch for quick convergence monitoring
        acc_list.append(accuracy(pred, batch_y))
        y_list.append(batch_y)
        pred_list.append(pred)

        # Backward pass and parameter update
        loss.backward()
        for opt in optimizer:
            opt.step()

        bce_total_loss += loss.item()

    # Concatenate all predictions and labels for epoch-level metrics
    y = torch.cat(y_list)
    pred = torch.cat(pred_list)

    auroc, auprc = roc_auc_cuda(y, pred)

    return bce_total_loss / batch_num, np.mean(acc_list), auroc, auprc


def eval_epoch(args, net, classifier_model, m_emb, g, hg_pos, hg_neg, hg_bigg, loss_f, validation_data, batch_size):
    net.eval()
    classifier_model.eval()

    valid_edges_set, valid_weight_set = validation_data

    # Pad validation hyperedges
    max_len = max(len(edge) for edges in [valid_edges_set] for edge in edges)
    valid_edges_padded = [edge + [0] * (max_len - len(edge)) for edge in valid_edges_set]
    valid_edges_set = torch.tensor(valid_edges_padded, dtype=torch.long, device=device)
    valid_weight_set = torch.tensor(valid_weight_set, device=device)

    with torch.no_grad():
        score_list = []

        # Prediction loop over all candidate reactions
        for i in tqdm(range(int(math.floor(len(valid_edges_set) / batch_size))),
                      mininterval=0.1, desc='  - (Prediction)   ', leave=True, ncols=100):
            batch_x = valid_edges_set[i * batch_size:(i + 1) * batch_size]
            batch_w = valid_weight_set[i * batch_size:(i + 1) * batch_size]

            # Re-compute upstream embeddings for each batch (ensures fresh representations)
            X1, X2, Y_pos, Y_neg = net(m_emb, g, hg_bigg, hg_bigg)
            if X1.dim() > 2:
                X1 = X1.squeeze()
            classifier_model.set_node_embedding(X1)

            # Classifier forward pass (only positive candidates during inference)
            pred_batch = classifier_model(batch_x, return_recon=False)
            pred_batch = pred_batch.squeeze(1)

            score_list.append(pred_batch)

        scores = torch.cat(score_list).float()

    return scores


def train(args, net, classifier_model, m_emb, g, hg_pos, hg_neg, hg_bigg, metabolite_count, reaction_count, loss_f,
          training_data, validation_data, optimizer, epochs, batch_size):
    logger = logging.getLogger(__name__)

    # Track best model based on validation AUROC

    best_metrics = {
        'epoch': -1,
        'bce_loss': float('inf'),
        'auroc': -1.0
    }
    best_model_path = os.path.join(args.save_path, 'best_model.pth')

    for epoch_i in tqdm(range(epochs), desc="Training", mininterval=0.1, leave=True, ncols=100):
        bce_loss, train_acc, auroc, auprc = train_epoch(
            args, net, classifier_model, m_emb, g, hg_pos, hg_neg, hg_bigg,
            metabolite_count, reaction_count, loss_f, training_data, optimizer, batch_size)

        # Save checkpoint whenever validation AUROC improves
        # Reason: Ensures the best-performing model is preserved for downstream prediction
        if auroc > best_metrics['auroc']:
            best_metrics = {
                'epoch': epoch_i,
                'bce_loss': bce_loss,
                'auroc': auroc,
                'auprc': auprc
            }
            checkpoint = {
                'net': net.state_dict(),
                'classifier_model': classifier_model.state_dict(),
                'epoch': epoch_i,
                'metrics': best_metrics
            }
            torch.save(checkpoint, best_model_path)

    torch.cuda.empty_cache()

    return net, classifier_model, best_metrics



