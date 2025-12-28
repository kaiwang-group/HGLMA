from typing import Tuple, Any
import torch
import torch.nn as nn
import dhg
from dgl.nn.pytorch.conv import DGNConv
import numpy as np

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self, in_channels: int,
                 hid_channels: int,
                 out_channels: int,
                 num_dgnn: int,
                 num_hgnn: int,
                 use_bn: bool = False) -> None:
        super().__init__()
        self.mapping_layer = feature_mapping_mlp(2816, in_channels)
        self.layers = nn.ModuleList()
        self.num_dgnn = num_dgnn
        self.num_hgnn = num_hgnn
        if self.num_dgnn == 1:
            self.layers.append(
                DGNConv(in_channels, hid_channels, ['dir2-av', 'dir2-dx', 'sum'], ['identity', 'amplification'], 2.5))
        else:
            # input layer
            self.layers.append(
                DGNConv(in_channels, hid_channels, ['dir3-av', 'dir3-dx', 'sum'], ['identity', 'amplification'], 2.5))
            # hidden layer
            for i in range(self.num_dgnn - 1):
                self.layers.append(
                    DGNConv(hid_channels, hid_channels, ['dir3-av', 'dir3-dx', 'sum'], ['identity', 'amplification'],
                            2.5))

        if self.num_hgnn == 1:
            self.layers.append(HGNN(hid_channels, out_channels, use_bn=use_bn, is_last=True))
        else:
            for i in range(self.num_hgnn - 1):
                self.layers.append(HGNN(hid_channels, hid_channels, use_bn=use_bn))
            self.layers.append(HGNN(hid_channels, out_channels, use_bn=use_bn, is_last=True))

    def forward(self, m_emb: torch.Tensor, g, hg_pos: dhg.Hypergraph, hg_neg: dhg.Hypergraph) -> Any:
        mapping_features = self.mapping_layer(m_emb)
        for i in range(self.num_dgnn):
            mapping_features = self.layers[i](g, mapping_features, eig_vec=g.ndata['eig'])
        X1 = mapping_features
        X2 = mapping_features
        for i in range(self.num_dgnn, self.num_dgnn + self.num_hgnn):
            X1, Y_pos = self.layers[i](X1, hg_pos)
            X2, Y_neg = self.layers[i](X2, hg_neg)
        return X1, X2, Y_pos, Y_neg


class HGNN(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            bias: bool = True,
            use_bn: bool = False,
            drop_rate: float = 0.5,
            is_last: bool = False,
    ):
        super().__init__()
        self.is_last = is_last
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, X: torch.Tensor, hg: dhg.Hypergraph) -> Tuple[Any, Any]:
        X = self.theta(X)
        if self.bn is not None:
            X = self.bn(X)
        Y = hg.v2e(X, aggr="mean")
        X = hg.e2v(Y, aggr="mean")
        if not self.is_last:
            X = self.drop(self.act(X))
        return X, Y


class Classifier(nn.Module):
    def __init__(
            self,
            n_head,
            d_model,
            d_k,
            d_v,
            node_embedding,
            metabolite_count,
            diag_mask,
            bottle_neck,
            **args):
        super().__init__()

        # Determine node_embedding type
        self.node_embedding = node_embedding
        self.is_tensor_embedding = isinstance(node_embedding, torch.Tensor)
        if self.is_tensor_embedding:
            self.node_embedding = self.node_embedding.to(args.get('device', 'cuda'))
        elif node_embedding is None:
            n_nodes = metabolite_count
            self.node_embedding = torch.randn(n_nodes, bottle_neck).to(args.get('device', 'cuda'))
            self.is_tensor_embedding = True  # Default to tensor

        self.pff_classifier = PositionwiseFeedForward(
            [d_model, 1], reshape=True, use_bias=True)

        self.encode1 = EncoderLayer(
            n_head,
            d_model,
            d_k,
            d_v,
            dropout_mul=0.3,
            dropout_pff=0.4,
            diag_mask=diag_mask,
            bottle_neck=bottle_neck)
        self.diag_mask_flag = diag_mask
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def set_node_embedding(self, new_embedding):
        """Dynamically update node_embedding"""
        self.node_embedding = new_embedding
        self.is_tensor_embedding = isinstance(new_embedding, torch.Tensor)

    def get_node_embeddings(self, x, return_recon=False):
        # shape of x: (b, tuple)
        sz_b, len_seq = x.shape
        x_flat = x.view(-1)  # (b * seq_len,)

        if self.is_tensor_embedding:
            # If node_embedding is tensor, index directly
            embedded_x = self.node_embedding[x_flat]  # (b * seq_len, bottle_neck)
            recon_loss = torch.tensor(0.0).to(x.device)  # No reconstruction loss
        else:
            # If node_embedding is callable (e.g. nn.Embedding)
            embedded_x, recon_loss = self.node_embedding(x_flat)

        if return_recon:
            return embedded_x.view(sz_b, len_seq, -1), recon_loss
        else:
            return embedded_x.view(sz_b, len_seq, -1)

    def get_embedding(self, x, slf_attn_mask, non_pad_mask, return_recon=False):
        if return_recon:
            x, recon_loss = self.get_node_embeddings(x, return_recon)
        else:
            x = self.get_node_embeddings(x, return_recon)
        dynamic, static, attn = self.encode1(x, x, slf_attn_mask, non_pad_mask)
        if return_recon:
            return dynamic, static, attn, recon_loss
        else:
            return dynamic, static, attn

    def get_embedding_static(self, x):
        if len(x.shape) == 1:
            x = x.view(-1, 1)
            flag = True
        else:
            flag = False
        slf_attn_mask = get_attn_key_pad_mask(seq_k=x, seq_q=x)
        non_pad_mask = get_non_pad_mask(x)
        x = self.get_node_embeddings(x)
        dynamic, static, attn = self.encode1(x, x, slf_attn_mask, non_pad_mask)
        if flag:
            return static[:, 0, :]
        return static

    def forward(self, x, mask=None, get_outlier=None, return_recon=False):
        x = x.long()

        slf_attn_mask = get_attn_key_pad_mask(seq_k=x, seq_q=x)
        non_pad_mask = get_non_pad_mask(x)

        dynamic, static, attn = self.get_embedding(x, slf_attn_mask, non_pad_mask)
        dynamic = self.layer_norm1(dynamic)
        static = self.layer_norm2(static)
        sz_b, len_seq, dim = dynamic.shape

        if self.diag_mask_flag == 'True':
            output = (dynamic - static) ** 2
        else:
            output = dynamic

        output = self.pff_classifier(output)
        output = torch.sigmoid(output)

        if get_outlier is not None:
            k = get_outlier
            outlier = (
                    (1 -
                     output) *
                    non_pad_mask).topk(
                k,
                dim=1,
                largest=True,
                sorted=True)[1]
            return outlier.view(-1, k)

        mode = 'sum'

        if mode == 'min':
            output, _ = torch.max(
                (1 - output) * non_pad_mask, dim=-2, keepdim=False)
            output = 1 - output

        elif mode == 'sum':
            output = torch.sum(output * non_pad_mask, dim=-2, keepdim=False)
            mask_sum = torch.sum(non_pad_mask, dim=-2, keepdim=False)
            output = output / (mask_sum + 1e-10)  # Add epsilon to avoid div by zero
        elif mode == 'first':
            output = output[:, 0, :]

        return output


class PositionwiseFeedForward(nn.Module):
    def __init__(
            self,
            dims,
            dropout=None,
            reshape=False,
            use_bias=True,
            residual=False,
            layer_norm=False):
        super(PositionwiseFeedForward, self).__init__()
        self.w_stack = []
        self.dims = dims
        for i in range(len(dims) - 1):
            self.w_stack.append(nn.Conv1d(dims[i], dims[i + 1], 1, bias=use_bias))
            self.add_module("PWF_Conv%d" % (i), self.w_stack[-1])
        self.reshape = reshape
        self.layer_norm = nn.LayerNorm(dims[-1])

        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.residual = residual
        self.layer_norm_flag = layer_norm

    def forward(self, x):
        output = x.transpose(1, 2)

        for i in range(len(self.w_stack) - 1):
            output = self.w_stack[i](output)
            output = torch.tanh(output)
            if self.dropout is not None:
                output = self.dropout(output)

        output = self.w_stack[-1](output)
        output = output.transpose(1, 2)

        if self.reshape:
            output = output.view(output.shape[0], -1, 1)

        if self.dims[0] == self.dims[-1]:
            # residual
            if self.residual:
                output = output + x

            if self.layer_norm_flag:
                output = self.layer_norm(output)

        return output


class FeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, dims, dropout=None, reshape=False, use_bias=True):
        super(FeedForward, self).__init__()
        self.w_stack = []
        for i in range(len(dims) - 1):
            self.w_stack.append(nn.Linear(dims[i], dims[i + 1], use_bias))
            self.add_module("FF_Linear%d" % (i), self.w_stack[-1])

        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.reshape = reshape

    def forward(self, x):
        output = x
        for i in range(len(self.w_stack) - 1):
            output = self.w_stack[i](output)
            output = torch.tanh(output)
            if self.dropout is not None:
                output = self.dropout(output)
        output = self.w_stack[-1](output)

        if self.reshape:
            output = output.view(output.shape[0], -1, 1)

        return output


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def masked_softmax(self, vector: torch.Tensor,
                       mask: torch.Tensor,
                       dim: int = -1,
                       memory_efficient: bool = False,
                       mask_fill_value: float = -1e32) -> torch.Tensor:

        if mask is None:
            result = torch.nn.functional.softmax(vector, dim=dim)
        else:
            mask = mask.float()
            while mask.dim() < vector.dim():
                mask = mask.unsqueeze(1)
            if not memory_efficient:
                # To limit numerical errors from large vector elements outside
                # the mask, we zero these out.
                result = torch.nn.functional.softmax(vector * mask, dim=dim)
                result = result * mask
                result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
            else:
                masked_vector = vector.masked_fill(
                    (1 - mask).bool(), mask_fill_value)
                result = torch.nn.functional.softmax(masked_vector, dim=dim)
        return result

    def forward(self, q, k, v, diag_mask, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -float('inf'))

        attn = self.masked_softmax(
            attn, diag_mask, dim=-1, memory_efficient=True)

        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(
            self,
            n_head,
            d_model,
            d_k,
            d_v,
            dropout,
            diag_mask,
            input_dim):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(input_dim, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(input_dim, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(input_dim, n_head * d_v, bias=False)

        nn.init.normal_(self.w_qs.weight, mean=0,
                        std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0,
                        std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0,
                        std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(
            temperature=np.power(d_k, 0.5))

        self.fc1 = FeedForward([n_head * d_v, d_model], use_bias=False)
        self.fc2 = FeedForward([n_head * d_v, d_model], use_bias=False)

        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.layer_norm3 = nn.LayerNorm(input_dim)

        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = dropout

        self.diag_mask_flag = diag_mask
        self.diag_mask = None

    def pass_(self, inputs):
        return inputs

    def forward(self, q, k, v, diag_mask, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        q = self.layer_norm1(q)
        k = self.layer_norm2(k)
        v = self.layer_norm3(v)

        sz_b, len_q, _ = q.shape
        sz_b, len_k, _ = k.shape
        sz_b, len_v, _ = v.shape

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous(
        ).view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous(
        ).view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous(
        ).view(-1, len_v, d_v)  # (n*b) x lv x dv

        n = sz_b * n_head

        if self.diag_mask is not None:
            if (len(self.diag_mask) <= n) or (
                    self.diag_mask.shape[1] != len_v):
                self.diag_mask = torch.ones((len_v, len_v), device=q.device)
                if self.diag_mask_flag == 'True':
                    self.diag_mask -= torch.eye(len_v, len_v, device=q.device)
                self.diag_mask = self.diag_mask.repeat(n, 1, 1)
                diag_mask = self.diag_mask
            else:
                diag_mask = self.diag_mask[:n]

        else:
            self.diag_mask = (torch.ones((len_v, len_v), device=q.device))
            if self.diag_mask_flag == 'True':
                self.diag_mask -= torch.eye(len_v, len_v, device=q.device)
            self.diag_mask = self.diag_mask.repeat(n, 1, 1)
            diag_mask = self.diag_mask

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..

        dynamic, attn = self.attention(q, k, v, diag_mask, mask=mask)

        dynamic = dynamic.view(n_head, sz_b, len_q, d_v)
        dynamic = dynamic.permute(
            1, 2, 0, 3).contiguous().view(
            sz_b, len_q, -1)  # b x lq x (n*dv)
        static = v.view(n_head, sz_b, len_q, d_v)
        static = static.permute(
            1, 2, 0, 3).contiguous().view(
            sz_b, len_q, -1)  # b x lq x (n*dv)

        dynamic = self.dropout(self.fc1(dynamic)) if self.dropout is not None else self.fc1(dynamic)
        static = self.dropout(self.fc2(static)) if self.dropout is not None else self.fc2(static)

        return dynamic, static, attn


class EncoderLayer(nn.Module):
    '''A self-attention layer + 2 layered pff'''

    def __init__(
            self,
            n_head,
            d_model,
            d_k,
            d_v,
            dropout_mul,
            dropout_pff,
            diag_mask,
            bottle_neck):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.mul_head_attn = MultiHeadAttention(
            n_head,
            d_model,
            d_k,
            d_v,
            dropout=dropout_mul,
            diag_mask=diag_mask,
            input_dim=bottle_neck)
        self.pff_n1 = PositionwiseFeedForward(
            [d_model, d_model, d_model], dropout=dropout_pff, residual=True, layer_norm=True)
        self.pff_n2 = PositionwiseFeedForward(
            [bottle_neck, d_model, d_model], dropout=dropout_pff, residual=False, layer_norm=True)

    def forward(self, dynamic, static, slf_attn_mask, non_pad_mask):
        dynamic, static1, attn = self.mul_head_attn(
            dynamic, dynamic, static, slf_attn_mask)
        dynamic = self.pff_n1(dynamic * non_pad_mask) * non_pad_mask
        static1 = self.pff_n2(static * non_pad_mask) * non_pad_mask

        return dynamic, static1, attn


def feature_mapping_mlp(in_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, out_dim)
    )


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(0).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(0)
    padding_mask = padding_mask.unsqueeze(
        1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask