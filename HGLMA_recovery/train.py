import logging
from model import *
from utils import *
import torch
import math
import os
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import argparse
import time
from sklearn.model_selection import KFold
import glob
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def train_batch_hyperedge(classifier_model, loss_f, batch_edges_pos, batch_edge_weight_pos, batch_edges_neg,
                          batch_edge_weight_neg, batch_y_pos, batch_y_neg):
    device = next(classifier_model.parameters()).device
    x_pos = torch.tensor(batch_edges_pos, dtype=torch.long, device=device)
    x_neg = torch.tensor(batch_edges_neg, dtype=torch.long, device=device)
    w_pos = torch.tensor(batch_edge_weight_pos, device=device)
    w_neg = torch.tensor(batch_edge_weight_neg, device=device)

    if len(batch_y_pos) == 0:
        x = torch.cat([x_pos, x_neg])  # Concatenate positive and negative edges
        w = torch.cat([w_pos, w_neg])  # Concatenate weights
        # Concatenate labels
        y = torch.cat([torch.ones((len(x_pos)), device=device), torch.zeros((len(x_neg)), device=device)])

        index = torch.randperm(len(x), device=device)
        x, y, w = x[index], y[index], w[index]

    # Forward pass
    pred = classifier_model(x, return_recon=False)
    pred = pred.squeeze(1)  # Remove the second dimension
    loss = loss_f(pred, y, weight=w)
    return pred, y, loss


def train_epoch(args, net, classifier_model, m_emb, g, hg_pos, hg_neg, hg_bigg, train_set, metabolite_count,
                reaction_count, loss_f, training_data, optimizer, batch_size):
    net.train()
    classifier_model.train()

    edges_pos, edge_weight_pos, edges_neg, edge_weight_neg = training_data
    # Padding
    max_len = max(len(edge) for edge in edges_pos)
    edges_pos_padded_pos = [edge + [0] * (max_len - len(edge)) for edge in edges_pos]
    edges_pos_padded_neg = [edge + [0] * (max_len - len(edge)) for edge in edges_neg]
    edges_pos = torch.tensor(edges_pos_padded_pos, dtype=torch.long, device=device)
    edges_neg = torch.tensor(edges_pos_padded_neg, dtype=torch.long, device=device)

    # Shuffle
    index = torch.randperm(len(edges_pos)).numpy()
    edges_pos, edge_weight_pos = edges_pos[index], edge_weight_pos[index]
    edges_neg, edge_weight_neg = edges_neg[index], edge_weight_neg[index]

    bce_total_loss = 0
    acc_list, y_list, pred_list = [], [], []
    y_pos = torch.tensor([])
    y_neg = torch.tensor([])

    if len(y_pos) > 0:
        y_pos, y_neg = y_pos[index], y_neg[index]

    batch_num = int(math.floor(len(edges_pos) / batch_size))
    bar = trange(batch_num, mininterval=0.1, desc='  - (Training) ', leave=False, )
    for i in bar:
        # Zero gradients
        for opt in optimizer:
            opt.zero_grad()

        batch_edges_pos = edges_pos[i * batch_size:(i + 1) * batch_size]
        batch_edge_weight_pos = edge_weight_pos[i * batch_size:(i + 1) * batch_size]
        batch_edges_neg = edges_neg[i * batch_size:(i + 1) * batch_size]
        batch_edge_weight_neg = edge_weight_neg[i * batch_size:(i + 1) * batch_size]

        batch_y_pos = ""
        batch_y_neg = ""
        if len(y_pos) > 0:
            batch_y_pos = y_pos[i * batch_size:(i + 1) * batch_size]
            batch_y_neg = y_neg[i * batch_size:(i + 1) * batch_size]
            if len(batch_y_pos) == 0:
                continue

        # Upstream model forward (Embedding + Directed Graph + Hypergraph)
        X1, X2, Y_neg, Y_neg = net(m_emb, g, hg_bigg, hg_bigg)

        if X1.dim() > 2:
            X1 = X1.squeeze()
        classifier_model.set_node_embedding(X1)

        # Downstream model forward (SAGNN)
        pred, batch_y, loss = train_batch_hyperedge(classifier_model, loss_f, batch_edges_pos, batch_edge_weight_pos,
                                                    batch_edges_neg, batch_edge_weight_neg, batch_y_pos, batch_y_neg)

        acc_list.append(accuracy(pred, batch_y))
        y_list.append(batch_y)
        pred_list.append(pred)

        # Backward
        loss.backward()
        # Update parameters
        for opt in optimizer:
            opt.step()
        bar.set_description(" - (Training) BCE:  %.4f" % (bce_total_loss / (i + 1)))
        bce_total_loss = bce_total_loss + loss.item()

    y = torch.cat(y_list)
    pred = torch.cat(pred_list)
    auroc, auprc = roc_auc_cuda(y, pred)
    return bce_total_loss / batch_num, np.mean(acc_list), auroc, auprc


def eval_epoch(args, net, classifier_model, m_emb, g, hg_pos, hg_neg, hg_bigg, loss_f, validation_data, batch_size):
    bce_total_loss = 0

    net.eval()
    classifier_model.eval()

    valid_edges_pos_set, valid_weight_pos_set, valid_edges_neg_set, valid_weight_neg_set = validation_data

    # Padding
    max_len = max(len(edge) for edges in [valid_edges_pos_set, valid_edges_neg_set] for edge in edges)
    valid_edges_pos_padded = [edge + [0] * (max_len - len(edge)) for edge in valid_edges_pos_set]
    valid_edges_neg_padded = [edge + [0] * (max_len - len(edge)) for edge in valid_edges_neg_set]
    valid_edges_pos_set = torch.tensor(valid_edges_pos_padded, dtype=torch.long, device=device)
    valid_edges_neg_set = torch.tensor(valid_edges_neg_padded, dtype=torch.long, device=device)
    valid_weight_pos_set = torch.tensor(valid_weight_pos_set, device=device)
    valid_weight_neg_set = torch.tensor(valid_weight_neg_set, device=device)

    with torch.no_grad():
        index_pos = torch.randperm(len(valid_edges_pos_set), device=device)
        index_neg = torch.randperm(len(valid_edges_neg_set), device=device)
        valid_edges_pos_set, valid_weight_pos_set = valid_edges_pos_set[index_pos], valid_weight_pos_set[index_pos]
        valid_edges_neg_set, valid_weight_neg_set = valid_edges_neg_set[index_neg], valid_weight_neg_set[index_neg]

        label_list = []
        score_list = []
        ratio = max(1, len(valid_edges_neg_set) // len(valid_edges_pos_set))
        batch_size_neg = batch_size * ratio

        # Batch count
        batch_count = len(valid_edges_pos_set) // batch_size

        # Validation loop
        for i in tqdm(range(batch_count), mininterval=0.1, desc='  - (Validation)   ', leave=False):
            # Positive edges slice
            start_idx_pos = i * batch_size
            end_idx_pos = start_idx_pos + batch_size
            batch_x_pos = valid_edges_pos_set[start_idx_pos:end_idx_pos]
            batch_w_pos = valid_weight_pos_set[start_idx_pos:end_idx_pos]

            # Negative edges slice
            start_idx_neg = i * batch_size_neg
            end_idx_neg = start_idx_neg + batch_size_neg
            batch_x_neg = valid_edges_neg_set[start_idx_neg:end_idx_neg]
            batch_w_neg = valid_weight_neg_set[start_idx_neg:end_idx_neg]

            # Dynamic computation of net output per batch
            X1, X2, Y_pos, Y_neg = net(m_emb, g, hg_bigg, hg_bigg)
            if X1.dim() > 2:
                X1 = X1.squeeze()
            classifier_model.set_node_embedding(X1)

            batch_x = torch.cat([batch_x_pos, batch_x_neg])
            batch_w = torch.cat([batch_w_pos, batch_w_neg])
            batch_y = torch.cat([
                torch.ones(len(batch_x_pos), device=device),
                torch.zeros(len(batch_x_neg), device=device)])

            index = torch.randperm(len(batch_x))
            batch_x, batch_y, batch_w = batch_x[index], batch_y[index], batch_w[index]

            pred_batch = classifier_model(batch_x, return_recon=False)
            pred_batch = pred_batch.squeeze(1)

            # Compute loss
            loss = loss_f(pred_batch, batch_y, weight=batch_w)
            bce_total_loss += loss.item() * len(batch_x)

            # Collect results
            label_list.append(batch_y)
            score_list.append(pred_batch)

        # Merge all batch results
        label = torch.cat(label_list)
        scores = torch.cat(score_list)

        # Calculate positive ratio in top scores
        scores_with_labels = torch.stack([scores, label], dim=1)
        # Sort descending by score
        sorted_indices = torch.argsort(scores_with_labels[:, 0], descending=True, stable=True)
        sorted_scores_with_labels = scores_with_labels[sorted_indices]

        top_k = args.top
        top_indices = sorted_indices[:top_k]
        top_labels = label[top_indices]
        positive_ratio = torch.sum(top_labels == 1).float() / len(top_indices) if len(top_indices) > 0 else 0.0

    return bce_total_loss / len(label), positive_ratio


def train(args, net, classifier_model, m_emb, g, hg_pos, hg_neg, hg_bigg, train_set, valid_set, metabolite_count,
          reaction_count, loss_f, training_data, validation_data, optimizer, epochs, batch_size):
    logger = logging.getLogger(__name__)
    best_metrics = {
        'epoch': -1,
        'valid_bce_loss': float('inf'),
        'valid_positive_ratio': -1
    }

    for epoch_i in range(epochs):
        logger.info(f'[ Epoch {epoch_i} of {epochs} ]')

        bce_loss, train_acc, auroc, auprc = train_epoch(
            args, net, classifier_model, m_emb, g, hg_pos, hg_neg, hg_bigg, train_set, metabolite_count, reaction_count,
            loss_f, training_data, optimizer, batch_size)
        logger.info('  - (Training)   bce: {bce_loss: 7.4f} '
                    ' acc: {acc:3.3f} %, auroc: {auroc:3.3f}, auprc: {auprc:3.3f}'.format(
            bce_loss=bce_loss,
            acc=100 * train_acc,
            auroc=auroc,
            auprc=auprc))

        loss, positive_ratio = eval_epoch(args, net, classifier_model, m_emb, g, hg_pos, hg_neg, hg_bigg, loss_f,
                                          validation_data, batch_size)

        logger.info('  - (Validation) Epoch {epoch}: Loss: {loss:.4f}, Top-{top} Positive Ratio: {ratio:.4f}'.format(
            epoch=epoch_i,
            loss=loss,
            top=args.top,
            ratio=positive_ratio))

        # Update best model based on highest positive_ratio
        if positive_ratio >= best_metrics['valid_positive_ratio']:
            best_metrics = {
                'epoch': epoch_i,
                'valid_bce_loss': loss,
                'valid_positive_ratio': positive_ratio
            }

        checkpoint = {
            'net': net.state_dict(),
            'classifier_model': classifier_model.state_dict(),
            'epoch': epoch_i,
            'metrics': best_metrics
        }
        torch.save(checkpoint, os.path.join(args.save_path, 'last_model.pth'))

    torch.cuda.empty_cache()

    logger.info(
        '\n[Best Model (by Top-{top} Positive Ratio) Metrics at Epoch {epoch}]: Loss: {loss:.4f}, Top-{top} Positive Ratio: {ratio:.4f}'.format(
            top=args.top,
            epoch=best_metrics['epoch'],
            loss=best_metrics['valid_bce_loss'],
            ratio=best_metrics['valid_positive_ratio']
        ))

    return net, classifier_model, best_metrics