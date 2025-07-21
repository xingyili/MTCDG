import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, ChebConv
from torch.nn import Dropout, Linear
import torch_geometric
import numpy as np
from torch.nn import Sequential


class GCN_mgaev3(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN_mgaev3, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=False, add_self_loops=False))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=False, add_self_loops=False))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=False, add_self_loops=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xx.append(x)
        x = self.convs[-1](x, adj_t)
        xx.append(F.relu(x))
        return xx

    def outEmb(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xx.append(x)
        x = self.convs[-1](x, adj_t)
        xx.append(x)
        x = torch.cat(xx, dim=1)
        return xx


class SAGE_mgaev2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE_mgaev2, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def mask_decode(self, x):
        x = torch.cat([self.n_emb.weight, x], dim=-1)
        for lin in self.mask_lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.mask_lins[-1](x)
        return x

    def forward(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xx.append(x)
        x = self.convs[-1](x, adj_t)
        xx.append(F.relu(x))
        return xx

    def outEmb(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xx.append(x)
        x = self.convs[-1](x, adj_t)
        xx.append(x)
        x = torch.cat(xx, dim=1)
        return x


class GIN_mgaev2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, decoder_mask='nmask', eps: float = 0.,  bias=True, xavier=True):
        super(GIN_mgaev2, self).__init__()
        self.decoder_mask = decoder_mask
        self.initial_eps = eps
        self.convs = torch.nn.ModuleList()
        self.act = torch.nn.ReLU()
        for i in range(num_layers - 1):
            start_dim = hidden_channels if i else in_channels
            nn = Sequential(Linear(start_dim, hidden_channels, bias=bias),
                            self.act,
                            Linear(hidden_channels, hidden_channels, bias=bias))
            # if xavier:
            #     self.weights_init(nn)
            conv = GINConv(nn)
            self.convs.append(conv)
        nn = Sequential(Linear(hidden_channels, hidden_channels, bias=bias),
                        self.act,
                        Linear(hidden_channels, out_channels, bias=bias))
        # if xavier:
        #     self.weights_init(nn)
        conv = GINConv(nn)
        self.convs.append(conv)

        self.dropout = dropout

    def weights_init(self, module):
        for m in module.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def reset_parameters(self):
        for conv in self.convs:
            # self.weights_init(conv.nn)
            # conv.eps.data.fill_(self.initial_eps)
            conv.reset_parameters()

    def mask_decode(self, x):
        x = torch.cat([self.n_emb.weight, x], dim=-1)
        for lin in self.mask_lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.mask_lins[-1](x)
        return x

    def forward(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xx.append(x)
        x = self.convs[-1](x, adj_t)
        xx.append(F.relu(x))
        return xx

    def outEmb(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xx.append(x)
        x = self.convs[-1](x, adj_t)
        xx.append(x)
        x = torch.cat(xx, dim=1)
        return x


class LPDecoder_ogb(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, encoder_layer, num_layers,
                 dropout):
        super(LPDecoder_ogb, self).__init__()
        n_layer = encoder_layer * encoder_layer
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels * n_layer, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        # self.lins.append(torch.nn.Linear(in_channels * n_layer, in_channels * n_layer))
        # self.lins.append(torch.nn.Linear(in_channels * n_layer, hidden_channels))
        # self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def cross_layer(self, x_1, x_2):
        bi_layer = []
        for i in range(len(x_1)):
            xi = x_1[i]
            for j in range(len(x_2)):
                xj = x_2[j]
                bi_layer.append(torch.mul(xi, xj))
        bi_layer = torch.cat(bi_layer, dim=1)
        return bi_layer

    def forward(self, h, edge):
        src_x = [h[i][edge[0]] for i in range(len(h))]
        dst_x = [h[i][edge[1]] for i in range(len(h))]
        x = self.cross_layer(src_x, dst_x)
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


class MLP(torch.nn.Module):
    def __init__(self, in_channels, drop_rate, hidden_dim):
        super(MLP, self).__init__()
        # MLP
        HIDDEN_DIM = hidden_dim
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch_geometric.nn.dense.Linear(in_channels, HIDDEN_DIM,
                                weight_initializer="kaiming_uniform"))
        self.dropout = Dropout(drop_rate)
        self.lins.append(torch_geometric.nn.dense.Linear(HIDDEN_DIM, 1, weight_initializer="kaiming_uniform"))
    
    def reset_parameters(self):
        for conv in self.lins:
            conv.reset_parameters()

    def forward(self, x):
        x = self.lins[0](x).relu()
        x = self.dropout(x)
        x = self.lins[1](x)
        return torch.sigmoid(x)


class Shared_kernel(torch.nn.Module):
    def __init__(self, ppi_num):
        super(Shared_kernel,self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(torch.nn.Conv1d(ppi_num, 1, kernel_size=1, bias=True))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x_list):
        # stack按照维度进行堆叠，现在变成（13627，3，256），即每个顶点都有3个256维度的向量
        x = torch.stack(x_list, dim=1)  # (x,ppi_num,48)
        # 卷积之后relu，输入channel为3，输出channel为1
        x = F.relu(self.convs[0](x))  # (x,1,48)
        # 移除tensor中大小为1的那个维度
        x = x.squeeze()  # (x,48)
        return x
        