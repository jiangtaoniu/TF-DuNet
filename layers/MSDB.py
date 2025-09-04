import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, Tensor


class MSDB(nn.Module):
    """
    Multi-scale Spatial Dependency Block (MSDB).
    This block captures spatial dependencies using a dynamically learned graph structure.
    It learns an adaptive adjacency matrix and performs graph signal propagation to extract features.
    """
    def __init__(self, c_out, gd_model, conv_channel, skip_channel,
                 gcn_depth, dropout, propalpha, node_dim):
        super(MSDB, self).__init__()

        self.c_out = c_out
        self.conv_channel = conv_channel

        # 1. Node embeddings to learn the adaptive adjacency matrix
        self.nodevec1 = nn.Parameter(torch.randn(c_out, node_dim), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(node_dim, c_out), requires_grad=True)

        # 2. A flexible linear layer to map input features to the graph convolution dimension
        self.feature_mapper = nn.Linear(in_features=gd_model, out_features=conv_channel * c_out)

        # 3. Graph signal propagation layer
        self.gconv1 = mixprop(conv_channel, skip_channel, gcn_depth, dropout, propalpha)
        self.gelu = nn.GELU()

        # 4. Output projection layers using 1x1 convolutions for sequence length independence
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channel, out_channels=gd_model, kernel_size=(1, 1))
        self.end_conv_2 = nn.Conv2d(in_channels=gd_model, out_channels=gd_model, kernel_size=(c_out, 1))

        self.norm = nn.LayerNorm(gd_model)

    def forward(self, x):
        # Input shape of x: (B, T, gd_model)
        B, T, _ = x.shape

        # Learn the adaptive adjacency matrix
        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)

        # 1. Apply linear layer for feature mapping
        # Input: (B, T, gd_model) -> Output: (B, T, conv_channel * c_out)
        out = self.feature_mapper(x)

        # 2. Reshape and permute for graph convolution
        # (B, T, C_conv, N) -> (B, C_conv, N, T)
        out = out.view(B, T, self.conv_channel, self.c_out)
        out = out.permute(0, 2, 3, 1)

        # Perform graph signal propagation
        out = self.gelu(self.gconv1(out, adp))  # Shape: (B, skip_channel, c_out, T)

        # 3. Use new, length-independent convolution layers for output projection
        out = self.end_conv_1(out)  # Shape: (B, d_model, c_out, T)
        out = self.end_conv_2(out)  # Shape: (B, d_model, 1, T)

        # 4. Reshape to match input x for residual connection
        out = out.squeeze(2)      # Shape: (B, d_model, T)
        out = out.transpose(1, 2) # Shape: (B, T, d_model)

        return self.norm(x + out)


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncwl,vw->ncvl', (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out, bias=True):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias)

    def forward(self, x):
        return self.mlp(x)


class mixprop(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep + 1) * c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho = self.mlp(ho)
        return ho