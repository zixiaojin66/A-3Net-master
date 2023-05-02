import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GATConv, GCNConv, GINConv, RGCNConv
from torch_geometric.nn import global_max_pool, global_mean_pool, global_add_pool
from torch.nn import Parameter as Param
import numpy as np
from torch.nn.utils.weight_norm import weight_norm
import math
# GCN  model
class GCN(torch.nn.Module):
    def __init__(self, input_dim=78, input_dim_e=243, output_dim=64, output_dim_e=64):
        super(GCN, self).__init__()

        # graph layers : drug
        self.gcn1 = GCNConv(input_dim, 64)
        self.gcn2 = GCNConv(64, output_dim)
        self.fc_g1 = nn.Linear(output_dim, output_dim)
        self.fc_g2 = nn.Linear(output_dim, output_dim)

        # # graph layers : sideEffect
        self.gcn3 = GCNConv(input_dim_e, 128)
        self.gcn4 = GCNConv(128, output_dim)
        self.fc_g3 = nn.Linear(output_dim, output_dim)
        self.fc_g4 = nn.Linear(output_dim, output_dim)

        # activation and regularization
        self.relu = nn.ReLU()
        self.diag = DiagLayer(in_dim=output_dim)

    def forward(self, data, data_e, DF=False, not_FC=True):
        # graph input feed-forward
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x_e, edge_index_e = data_e.x, data_e.edge_index

        # 药物
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.gcn1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gcn2(x, edge_index)
        x = self.relu(x)
        x = global_max_pool(x, batch)  # global max pooling

        # 副作用
        x_e = F.dropout(x_e, p=0.2, training=self.training)
        x_e = self.gcn3(x_e, edge_index_e)
        x_e = self.relu(x_e)
        x_e = F.dropout(x_e, p=0.2, training=self.training)
        x_e = self.gcn4(x_e, edge_index_e)
        x_e = self.relu(x_e)

        if not not_FC:
            x = self.relu(self.fc_g1(x))
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.fc_g2(x)
            x_e = self.relu(self.fc_g3(x_e))
            x_e = F.dropout(x_e, p=0.5, training=self.training)
            x_e = self.fc_g4(x_e)

        # 结合
        x_ = self.diag(x) if DF else x

        xc = torch.matmul(x_, x_e.T)

        return xc, x, x_e


# GAT  model
class GAT(torch.nn.Module):
    def __init__(self, input_dim=109, input_dim_e=243, output_dim=200, output_dim_e=64, dropout=0.2, heads=10):
        super(GAT, self).__init__()

        # graph layers : drug
        self.gcn1 = GATConv(input_dim, 128, heads=heads, dropout=dropout)
        self.gcn2 = GATConv(128 * heads, output_dim, dropout=dropout)
        self.fc_g1 = nn.Linear(output_dim, output_dim)
        self.fc_g2 = nn.Linear(output_dim, output_dim)

        # # graph layers : sideEffect
        self.gcn3 = GATConv(input_dim_e, 128, heads=heads, dropout=dropout)
        self.gcn4 = GATConv(128 * heads, output_dim, dropout=dropout)
        self.fc_g3 = nn.Linear(output_dim, output_dim)
        self.fc_g4 = nn.Linear(output_dim, output_dim)

        # activation and regularization
        self.relu = nn.ReLU()
        self.diag = DiagLayer(in_dim=output_dim)

    def forward(self, data, data_e, DF=False, not_FC=True):
        # graph input feed-forward
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x_e, edge_index_e = data_e.x, data_e.edge_index
        # 药物
        x = self.relu(self.gcn1(x, edge_index))
        x = self.relu(self.gcn2(x, edge_index))
        x = global_max_pool(x, batch)  # global max pooling

        # 副作用
        x_e = self.gcn3(x_e, edge_index_e)
        x_e = self.relu(x_e)
        x_e = self.gcn4(x_e, edge_index_e)
        x_e = self.relu(x_e)

        if not not_FC:
            x = self.relu(self.fc_g1(x))
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.fc_g2(x)
            x_e = self.relu(self.fc_g3(x_e))
            x_e = F.dropout(x_e, p=0.5, training=self.training)
            x_e = self.fc_g4(x_e)

        # 结合
        x_ = self.diag(x) if DF else x

        xc = torch.matmul(x_, x_e.T)

        return xc, x, x_e


# GINConv model
class GIN(torch.nn.Module):

    def __init__(self, input_dim=78, input_dim_e=243, output_dim=64, dropout=0.2):
        super(GIN, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        # convolution layers
        nn1 = Sequential(Linear(input_dim, output_dim), ReLU(), Linear(output_dim, output_dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(output_dim)

        nn2 = Sequential(Linear(output_dim, output_dim), ReLU(), Linear(output_dim, output_dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(output_dim)

        nn3 = Sequential(Linear(output_dim, output_dim), ReLU(), Linear(output_dim, output_dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(output_dim)

        nn4 = Sequential(Linear(output_dim, output_dim), ReLU(), Linear(output_dim, output_dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(output_dim)

        nn5 = Sequential(Linear(output_dim, output_dim), ReLU(), Linear(output_dim, output_dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(output_dim)

        self.fc_g1 = nn.Linear(output_dim, output_dim)
        self.fc_g2 = nn.Linear(output_dim, output_dim)

        nn6 = Sequential(Linear(input_dim_e, output_dim), ReLU(), Linear(output_dim, output_dim))  # 时序容器。

        self.conv6 = GINConv(nn6)
        self.bn6 = torch.nn.BatchNorm1d(output_dim)

        nn7 = Sequential(Linear(output_dim, output_dim), ReLU(), Linear(output_dim, output_dim))
        self.conv7 = GINConv(nn7)
        self.bn7 = torch.nn.BatchNorm1d(output_dim)

        nn8 = Sequential(Linear(output_dim, output_dim), ReLU(), Linear(output_dim, output_dim))
        self.conv8 = GINConv(nn8)
        self.bn8 = torch.nn.BatchNorm1d(output_dim)

        nn9 = Sequential(Linear(output_dim, output_dim), ReLU(), Linear(output_dim, output_dim))
        self.conv9 = GINConv(nn9)
        self.bn9 = torch.nn.BatchNorm1d(output_dim)

        nn10 = Sequential(Linear(output_dim, output_dim), ReLU(), Linear(output_dim, output_dim))
        self.conv10 = GINConv(nn10)
        self.bn10 = torch.nn.BatchNorm1d(output_dim)

        self.fc_g3 = nn.Linear(output_dim, output_dim)
        self.fc_g4 = nn.Linear(output_dim, output_dim)

        self.diag = DiagLayer(in_dim=output_dim)
        # activation and regularization
        self.relu = nn.ReLU()

    def forward(self, data, data_e, DF=False, not_FC=True):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x_e, edge_index_e = data_e.x, data_e.edge_index

        # drug
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch)

        # side effect
        x_e = F.relu(self.conv6(x_e, edge_index_e))
        x_e = self.bn6(x_e)
        x_e = F.relu(self.conv7(x_e, edge_index_e))
        x_e = self.bn7(x_e)
        x_e = F.relu(self.conv8(x_e, edge_index_e))
        x_e = self.bn8(x_e)
        x_e = F.relu(self.conv9(x_e, edge_index_e))
        x_e = self.bn9(x_e)
        x_e = F.relu(self.conv10(x_e, edge_index_e))
        x_e = self.bn10(x_e)

        if not not_FC:
            x = self.relu(self.fc_g1(x))
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.fc_g2(x)
            x_e = self.relu(self.fc_g3(x_e))
            x_e = F.dropout(x_e, p=0.5, training=self.training)
            x_e = self.fc_g4(x_e)

        # 结合
        x_ = self.diag(x) if DF else x
        xc = torch.matmul(x_, x_e.T)

        return xc, x, x_e



# RGCN  model
class RGCN(torch.nn.Module):
    def __init__(self, input_dim=109, input_dim_e=243, output_dim=64, output_dim_e=64, dropout=0.2, heads=10):
        super(RGCN, self).__init__()

        # graph layers : drug
        self.gcn1 = RGCNConv(input_dim, 64, num_relations=5, num_bases=4, aggr='mean')
        self.gcn2 = RGCNConv(64, output_dim, num_relations=5, num_bases=4, aggr='mean')
        self.fc_g1 = nn.Linear(output_dim, output_dim)
        self.fc_g2 = nn.Linear(output_dim, output_dim)

        # # graph layers : sideEffect
        self.gcn3 = GATConv(input_dim_e, 128, heads=heads, dropout=dropout)
        self.gcn4 = GATConv(128 * heads, output_dim, dropout=dropout)
        self.fc_g3 = nn.Linear(output_dim, output_dim)
        self.fc_g4 = nn.Linear(output_dim, output_dim)

        # activation and regularization
        self.relu = nn.ReLU()
        self.diag = DiagLayer(in_dim=output_dim)

    def forward(self, data, data_e, DF=False, not_FC=True):
        # graph input feed-forward
        x, edge_index, edge_type, batch = data.x, data.edge_index, data.edge_type, data.batch
        x_e, edge_index_e = data_e.x, data_e.edge_index
        # print(x.shape)
        # 药物
        x = F.dropout(x, p=0.2, training=self.training)  # 将模型整体的training状态参数传入dropout函数
        x = torch.tanh(self.gcn1(x, edge_index, edge_type))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gcn2(x, edge_index, edge_type)
        x = torch.tanh(x)
        x = global_max_pool(x, batch)  # global max pooling

        # 副作用
        x_e = self.gcn3(x_e, edge_index_e)
        x_e = self.relu(x_e)
        x_e = self.gcn4(x_e, edge_index_e)
        x_e = self.relu(x_e)

        if not not_FC:
            x = self.relu(self.fc_g1(x))
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.fc_g2(x)
            x_e = self.relu(self.fc_g3(x_e))
            x_e = F.dropout(x_e, p=0.5, training=self.training)
            x_e = self.fc_g4(x_e)

        # 结合
        x_ = self.diag(x) if DF else x

        xc = torch.matmul(x_, x_e.T)

        return xc, x, x_e


class GAT1(torch.nn.Module):
    def __init__(self, input_dim=109, input_dim_e=243, output_dim=200, output_dim_e=64, dropout=0.2, heads=10):
        super(GAT1, self).__init__()

        # graph layers : drug
        self.gcn1 = GATConv(input_dim, output_dim, dropout=dropout)
        self.fc_g1 = nn.Linear(output_dim, output_dim)
        self.fc_g2 = nn.Linear(output_dim, output_dim)

        # # graph layers : sideEffect
        self.gcn3 = GATConv(input_dim_e, output_dim, dropout=dropout)
        self.fc_g3 = nn.Linear(output_dim, output_dim)
        self.fc_g4 = nn.Linear(output_dim, output_dim)

        # activation and regularization
        self.relu = nn.ReLU()
        self.diag = DiagLayer(in_dim=output_dim)

    def forward(self, data, data_e, DF=False, not_FC=True):
        # graph input feed-forward
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x_e, edge_index_e = data_e.x, data_e.edge_index
        # 药物
        x = self.relu(self.gcn1(x, edge_index))
        x = global_max_pool(x, batch)  # global max pooling

        # 副作用
        x_e = self.gcn3(x_e, edge_index_e)
        x_e = self.relu(x_e)

        if not not_FC:
            x = self.relu(self.fc_g1(x))
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.fc_g2(x)
            x_e = self.relu(self.fc_g3(x_e))
            x_e = F.dropout(x_e, p=0.5, training=self.training)
            x_e = self.fc_g4(x_e)

        # 结合
        x_ = self.diag(x) if DF else x
        xc = torch.matmul(x_, x_e.T)

        return xc, x, x_e

class DiagLayer(torch.nn.Module):
    def __init__(self, in_dim, num_et=1):
        super(DiagLayer, self).__init__()
        self.num_et = num_et
        self.in_dim = in_dim
        self.weight = Param(torch.Tensor(num_et, in_dim))

        self.reset_parameters()

    def forward(self, x):
        # print(self.weight)
        value = x * self.weight
        return value

    def reset_parameters(self):
        self.weight.data.normal_(std=1/np.sqrt(self.in_dim))
        # self.weight.data.fill_(1)

class A3_Net(torch.nn.Module):
    def __init__(self, input_dim=109, input_dim_e=243, output_dim=200, output_dim_e=64, dropout=0.2, heads=10):
        super(A3_Net, self).__init__()

        self.fc_1= nn.Linear(input_dim,output_dim)
        self.fc_2 = nn.Linear(input_dim_e, output_dim)
        self.att = nn.TransformerEncoderLayer(output_dim, 8)
        self.Att = nn.TransformerEncoder(self.att,num_layers=6)
        # graph layers : drug
        self.gcn1 = GATConv(input_dim, 128, heads=heads)
        self.gcn2 = GATConv(128 * heads, output_dim, heads=heads)
        self.gcn5 = GATConv(output_dim * heads, output_dim)
        self.fc_g1 = nn.Linear(output_dim, output_dim)
        self.fc_g2 = nn.Linear(output_dim, output_dim)

        # # graph layers : sideEffect
        self.gcn3 = GATConv(input_dim_e, 128, heads=heads)
        self.gcn4 = GATConv(128 * heads, output_dim, heads=heads)
        self.gcn6 = GATConv(output_dim * heads, output_dim)
        self.fc_g3 = nn.Linear(output_dim, output_dim)
        self.fc_g4 = nn.Linear(output_dim, output_dim)
        # activation and regularization
        self.relu = nn.ReLU()
        self.norm1 = nn.LayerNorm([input_dim])
        self.norm2 = nn.LayerNorm([input_dim_e])
        self.norm3 = nn.LayerNorm([200])
        self.norm4 = nn.LayerNorm([200])

        self.norm_1 = nn.LayerNorm([input_dim])
        self.norm_2 = nn.LayerNorm([1280])
        self.norm_3 = nn.LayerNorm([2000])

        self.norm_e_1 = nn.LayerNorm([input_dim_e])
        self.norm_e_2 = nn.LayerNorm([1280])
        self.norm_e_3 = nn.LayerNorm([2000])

        self.diag = DiagLayer(in_dim=output_dim)


    def forward(self, data, data_e, DF=False, not_FC=True,alpha=0.15):
        # graph input feed-forward
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x_e, edge_index_e = data_e.x, data_e.edge_index

        x = self.norm1(x)
        x_fc=self.relu(self.fc_1(x))
        x_e = self.norm2(x_e)
        x_e_fc=self.relu(self.fc_2(x_e))

        x_x_e = torch.cat((x_fc, x_e_fc), dim=0)
        x_x_e = self.relu(self.Att(x_x_e))
        drug_emb0, si_eff_emb0 = torch.split(x_x_e, [x_x_e.shape[0]-994, 994], dim=0)

        drug_emb0 = global_max_pool(drug_emb0, batch)

        # 药物
        x=self.norm_1(x)
        x = self.relu(self.gcn1(x, edge_index))
        x = self.norm_2(x)
        x = self.relu(self.gcn2(x, edge_index))
        x = self.norm_3(x)
        x = self.relu(self.gcn5(x, edge_index))
        x = global_max_pool(x, batch)  # global max pooling

        # 副作用
        x_e=self.norm_e_1(x_e)
        x_e = self.relu(self.gcn3(x_e, edge_index_e))
        x_e = self.norm_e_2(x_e)
        x_e = self.relu(self.gcn4(x_e, edge_index_e))
        x_e = self.norm_e_3(x_e)
        x_e = self.relu(self.gcn6(x_e, edge_index_e))

        x=(1-alpha)*x+alpha*drug_emb0
        x_e=(1-alpha*x_e)+alpha*si_eff_emb0

        if not not_FC:
            x=self.norm3(x)
            x = self.relu(self.fc_g1(x))
            x = self.fc_g2(x)

            x_e=self.norm4(x_e)
            x_e = self.relu(self.fc_g3(x_e))
            x_e = self.fc_g4(x_e)

        # 结合
        x_ = self.diag(x) if DF else x

        xc = torch.matmul(x_, x_e.T)

        return xc, x, x_e
