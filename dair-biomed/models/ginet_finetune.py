import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

num_atom_type = 119 # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 5 # including aromatic and self-loop edge
num_bond_direction = 3 


class MLP(nn.Module):
    def __init__(self, 
        task='classification', num_layer=5, emb_dim=512, drop_rate=0.3,
        kg_in_dim=64, text_in_dim=64, fp_in_dim=64, h_dim=512 
    ):
        super(MLP, self).__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.drop_rate = drop_rate
        self.text_feat_dim = text_in_dim + fp_in_dim+ h_dim
        self.kg_feat_dim = kg_in_dim + fp_in_dim + h_dim
        self.task = task

        if self.task == 'classification':
            out_dim = 2
        elif self.task == 'regression':
            out_dim = 1

        self.kg_input = nn.Linear(self.kg_feat_dim, self.emb_dim)
        self.text_input = nn.Linear(self.text_feat_dim, self.emb_dim)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=self.drop_rate)
        self.hidden = nn.Linear(self.emb_dim, 256)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(256, out_dim)
        
    def forward(self, h, fp_x, kg_x = None, text_x = None):
        if text_x is not None:
            x = torch.cat([h, fp_x, text_x], dim=-1)
            out = self.text_input(x)
        if kg_x is not None:
            # print(h.shape, fp_x.shape, kg_x.shape)
            x = torch.cat([h, fp_x, kg_x], dim=-1)
            # print(x.shape)
            out = self.kg_input(x)
        out = self.relu(out)
        out = self.drop(out)
        out = self.hidden(out)
        out = self.relu2(out)
        out = self.output(out)
        
        return x, out

class GINEConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GINEConv, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2*emb_dim), 
            nn.ReLU(), 
            nn.Linear(2*emb_dim, emb_dim)
        )
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + \
            self.edge_embedding2(edge_attr[:,1])

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GINet(nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat
    Output:
        node representations
    """
    def __init__(self, 
        task='classification', mod=0, num_layer=5, emb_dim=300, feat_dim=512, 
        drop_ratio=0, pool='mean', pred_n_layer=2, pred_act='softplus',
        kg_in_dim=256, text_in_dim=768, fp_in_dim=1024, ex_feat_out_dim=64 
    ):
        super(GINet, self).__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.kg_in_dim = kg_in_dim
        self.text_in_dim = text_in_dim
        self.fp_in_dim = fp_in_dim
        self.ex_feat_out_dim = ex_feat_out_dim
        self.drop_ratio = drop_ratio
        self.task = task
        self.mod = mod

        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        # List of MLPs
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GINEConv(emb_dim))

        # List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'max':
            self.pool = global_max_pool
        elif pool == 'add':
            self.pool = global_add_pool
        self.feat_lin = nn.Linear(self.emb_dim, self.feat_dim)
        self.kg_lin = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.kg_in_dim, self.ex_feat_out_dim)
        )
        self.text_lin = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.text_in_dim, self.ex_feat_out_dim)
        )
        self.fp_lin = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.fp_in_dim, self.ex_feat_out_dim)
        )

        if self.task == 'classification':
            out_dim = 2
        elif self.task == 'regression':
            out_dim = 1
        
        self.pred_n_layer = max(1, pred_n_layer)
        self.feat_dim += self.ex_feat_out_dim


        if pred_act == 'relu':
            pred_head = [
                nn.Linear(self.feat_dim, self.feat_dim//2), 
                nn.ReLU(inplace=True)
            ]
            for _ in range(self.pred_n_layer - 1):
                pred_head.extend([
                    nn.Linear(self.feat_dim//2, self.feat_dim//2), 
                    nn.ReLU(inplace=True),
                ])
            pred_head.append(nn.Linear(self.feat_dim//2, out_dim))
        elif pred_act == 'softplus':
            pred_head = [
                nn.Linear(self.feat_dim, self.feat_dim//2), 
                nn.Softplus()
            ]
            for _ in range(self.pred_n_layer - 1):
                pred_head.extend([
                    nn.Linear(self.feat_dim//2, self.feat_dim//2), 
                    nn.Softplus()
                ])
            pred_head.append(nn.Linear(self.feat_dim//2, out_dim))
        else:
            raise ValueError('Undefined activation function')
        
        self.pred_head = nn.Sequential(*pred_head)
        # self.mlp = MLP(drop_rate=drop_ratio)

        # print('kg_txet=========',mod)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        
        h = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])

        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

        h = self.pool(h, data.batch)
        h = self.feat_lin(h)
        kg_x = self.kg_lin(data.kg_x.view(-1, self.kg_in_dim))
        text_x = self.text_lin(data.text_x.view(-1, self.text_in_dim))
        fp_x = self.fp_lin(data.fp_x.view(-1, self.fp_in_dim))


        if self.mod == 1:
            return h, self.pred_head(torch.cat([h, kg_x], dim=-1))
        else:
            return h, self.pred_head(torch.cat([h, text_x], dim=-1))
        

        
        # return h, self.pred_head(torch.cat([h, fp_x, text_x], dim=-1))

        # return h, self.pred_head(torch.cat([h, fp_x, kg_x, text_x], dim=-1))
        # return h, self.pred_head(h)

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.parameter.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)
