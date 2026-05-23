import torch
import torch.nn as nn
import torch.nn.functional as F


class NConv(nn.Module):
    def __init__(self):
        super(NConv, self).__init__()

    def forward(self, x, adj_matrix):
        x = torch.einsum('ncwl,vw->ncvl', (x, adj_matrix))
        return x.contiguous()


class LinearLayer(nn.Module):
    def __init__(self, c_in, c_out, bias=True):
        super(LinearLayer, self).__init__()
        self.mlp = torch.nn.Conv2d(
            c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias
        )

    def forward(self, x):
        return self.mlp(x)


class MixProp(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(MixProp, self).__init__()
        self.n_conv = NConv()
        self.mlp = LinearLayer((gdep + 1) * c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d_nodes = adj.sum(1)
        h = x
        out = [h]
        a_matrix = adj / d_nodes.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.n_conv(h, a_matrix)
            out.append(h)
        h_out = torch.cat(out, dim=1)
        h_out = self.mlp(h_out)
        return h_out


class MSDB(nn.Module):
    """
    Multi-scale Spatial Dependency Block (MSDB).

    This block captures spatial dependencies using a dynamically learned graph structure.
    It learns an adaptive adjacency matrix and performs graph signal propagation (GCN) to
    extract local and global relational features across distinct scales.

    Attributes:
        c_out (int): Output feature size (number of nodes).
        d_model (int): Hidden dimension size of model.
        conv_channel (int): Intermediate channel size for mixing convolutions.
        skip_channel (int): Number of channels used across skip connections.
        nodevec1/nodevec2 (nn.Parameter): Trainable embeddings to derive the adaptive adjacency matrix.
    """

    def __init__(self, c_out, d_model, conv_channel, skip_channel,
                 gcn_depth, dropout, propalpha, node_dim):
        super(MSDB, self).__init__()

        self.c_out = c_out
        self.conv_channel = conv_channel

        self.nodevec1 = nn.Parameter(torch.randn(c_out, node_dim), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(node_dim, c_out), requires_grad=True)

        self.feature_mapper = nn.Linear(in_features=d_model, out_features=conv_channel * c_out)

        self.gconv1 = MixProp(conv_channel, skip_channel, gcn_depth, dropout, propalpha)
        self.gelu = nn.GELU()

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channel, out_channels=d_model, kernel_size=(1, 1))
        self.end_conv_2 = nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=(c_out, 1))

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        batch_size, t_steps, _ = x.shape

        # Construct dynamic, learned spatial dependency (adjacency) matrix via dot product of node vectors
        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)

        # Map embedding space into dense graph convolutions channel space
        out = self.feature_mapper(x)

        # Reshape to facilitate 2D convolutions over sequences and channel interactions:
        # (batch_size, time_steps, conv_channel, nodes) -> (batch_size, conv_channel, nodes, time_steps)
        out = out.view(batch_size, t_steps, self.conv_channel, self.c_out)
        out = out.permute(0, 2, 3, 1)

        # Execute mix-propagation logic powered by the learned graph topology
        out = self.gelu(self.gconv1(out, adp))

        # Project output channels back to standard model embeddings
        out = self.end_conv_1(out)
        out = self.end_conv_2(out)

        # Collapse the flattened spatial dimension and permute to align with main branch sequence logic: (Batch, Time, D_model)
        out = out.squeeze(2)
        out = out.transpose(1, 2)

        # Re-introduce original temporal feature context to graph enhanced representations via residual connection
        return self.norm(x + out)
