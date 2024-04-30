import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Linear, Sequential, ReLU
from torch_geometric.nn import global_mean_pool
from torch.nn.modules.batchnorm import _BatchNorm
import torch_geometric.nn as gnn
from torch import Tensor
from collections import OrderedDict


class NodeLevelBatchNorm(_BatchNorm):
    r"""
    Applies Batch Normalization over a batch of graph data.
    Shape:
        - Input: [batch_nodes_dim, node_feature_dim]
        - Output: [batch_nodes_dim, node_feature_dim]
    batch_nodes_dim: all nodes of a batch graph
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(NodeLevelBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def _check_input_dim(self, input):
        if input.dim() != 2:
            raise ValueError('expected 2D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        return torch.functional.F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)

    def extra_repr(self):
        return 'num_features={num_features}, eps={eps}, ' \
               'affine={affine}'.format(**self.__dict__)

class GraphConvBn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = gnn.GraphConv(in_channels, out_channels)
        self.norm = NodeLevelBatchNorm(out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        data.x = F.relu(self.norm(self.conv(x, edge_index)))

        return data
    
class GraphIsoBn(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.nn = Sequential(
                    Linear(in_channels, out_channels), ReLU(),
                    Linear(out_channels, out_channels)
        )
        self.conv = gnn.GINConv(self.nn)
        self.bn = nn.BatchNorm1d(out_channels)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv(x, edge_index))
        data.x = self.bn(x)

        return data
    
class GraphAttBn(nn.Module):
    def __init__(self, in_channels, out_channels, heads, dropout, activation) -> None:
        super().__init__()

        # graph layers
        self.conv = gnn.GATConv(in_channels, out_channels, heads, dropout)
        self.act = nn.ELU() if activation=='elu' else nn.ReLU()
        self.bn = nn.BatchNorm1d(out_channels*heads)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.act(self.conv(x, edge_index))
        data.x = self.bn(x)

        return data
    

# GCN based model
class GraphDTA_GCN(torch.nn.Module):
    def __init__(
        self, 
        n_output = 1,
        n_filters = 32,
        embed_dim = 128,
        num_features_xd = 78,
        num_features_xt = 25,
        output_dim = 128,
        dropout=0.2
    ):
        super(GraphDTA_GCN, self).__init__()
        self.n_output = n_output

        # drug molecule branch
        self.features_drug = nn.Sequential(OrderedDict([('embed', GraphConvBn(18, num_features_xd))]))
        ligand_encoder = nn.Sequential(
            GraphConvBn(num_features_xd, num_features_xd),
            GraphConvBn(num_features_xd, num_features_xd*2),
            GraphConvBn(num_features_xd*2, num_features_xd*4))
        self.features_drug.add_module('Encoder1', ligand_encoder)
        self.flatten = nn.Sequential(
            torch.nn.Linear(num_features_xd*4, 1024), ReLU(), nn.Dropout(dropout),
            torch.nn.Linear(1024, output_dim), nn.Dropout(dropout))

        # protein sequence branch (1d conv)
        self.features_target = Sequential(OrderedDict([('embed', nn.Embedding(num_features_xt + 1, embed_dim))]))
        protein_encoder = Sequential(
            nn.Conv1d(in_channels=1200, out_channels=n_filters, kernel_size=8), ReLU(),
            nn.Conv1d(in_channels=n_filters, out_channels=n_filters*2, kernel_size=8), ReLU(),
            # nn.Conv1d(in_channels=n_filters*2, out_channels=n_filters*4, kernel_size=8), ReLU(),
            )
        flatten_protein = Sequential(nn.Flatten(), Linear(32*2*114, output_dim))
        self.features_target.add_module('Encoder2', protein_encoder)
        self.features_target.add_module('Flatten', flatten_protein)

        # combined layers
        self.output = nn.Sequential(
            nn.Linear(2*output_dim, 1024), ReLU(), nn.Dropout(dropout),
            nn.Linear(1024, 512), ReLU(), nn.Dropout(dropout),
            nn.Linear(512, self.n_output))

    def forward(self, data):
        # drug encode
        drug_x = self.features_drug(data)
        drug_x = gnn.global_max_pool(drug_x.x, drug_x.batch)
        drug_x = self.flatten(drug_x)

        # target encode
        target = data.target
        target_x = self.features_target(target)

        # concat
        concat_x = torch.cat((drug_x, target_x), 1)
        out = self.output(concat_x)
        return out
    

# GINConv model
class GraphDTA_GIN(torch.nn.Module):
    def __init__(
            self, 
            n_output=1,
            n_filters=32,
            num_features_xd=78,
            num_features_xt=25,
            embed_dim=128,
            output_dim=128,
            dropout=0.2
        ):
        super(GraphDTA_GIN, self).__init__()

        dim = 32
        self.n_output = n_output

        # drug molecule branch
        self.features_drug = nn.Sequential(OrderedDict([('embed', GraphIsoBn(18, num_features_xd))]))
        ligand_encoder = nn.Sequential(
            GraphIsoBn(num_features_xd, dim), GraphIsoBn(dim, dim), GraphIsoBn(dim, dim),
            GraphIsoBn(dim, dim), GraphIsoBn(dim, dim))
        self.features_drug.add_module('Encoder1', ligand_encoder)
        self.flatten = nn.Sequential(
            torch.nn.Linear(dim, 256), ReLU(), nn.Dropout(dropout),
            torch.nn.Linear(256, output_dim), nn.Dropout(dropout))

        # protein sequence branch (1d conv)
        self.features_target = Sequential(OrderedDict([('embed', nn.Embedding(num_features_xt + 1, embed_dim))]))
        protein_encoder = Sequential(
            nn.Conv1d(in_channels=1200, out_channels=n_filters, kernel_size=8), ReLU(),
            nn.Conv1d(in_channels=n_filters, out_channels=n_filters*2, kernel_size=8), ReLU(),
            )
        flatten_protein = Sequential(nn.Flatten(), Linear(32*2*114, output_dim))
        self.features_target.add_module('Encoder2', protein_encoder)
        self.features_target.add_module('Flatten', flatten_protein)

        # combined layers
        self.output = nn.Sequential(
            nn.Linear(2*output_dim, 1024), ReLU(), nn.Dropout(dropout),
            nn.Linear(1024, 256), ReLU(), nn.Dropout(dropout),
            nn.Linear(256, self.n_output)
        )

    def forward(self, data):
        # drug encode
        drug_x = self.features_drug(data)
        drug_x = gnn.global_add_pool(drug_x.x, drug_x.batch)
        drug_x = self.flatten(drug_x)

        # target encode
        target = data.target
        target_x = self.features_target(target)

        # concat
        concat_x = torch.cat((drug_x, target_x), 1)
        out = self.output(concat_x)
        return out


# GAT model
class GraphDTA_GAT(torch.nn.Module):
    def __init__(
            self,
            n_output=1,
            n_filters=32,
            num_features_xd=78,
            num_features_xt=25,
            embed_dim=128,
            output_dim=128,
            dropout=0.2
        ):
        super(GraphDTA_GAT, self).__init__()
        self.n_output = n_output

        # drug molecule branch
        self.features_drug = nn.Sequential(OrderedDict([('embed', GraphAttBn(18, num_features_xd, 1, dropout, 'elu'))]))
        ligand_encoder = nn.Sequential(
            GraphAttBn(num_features_xd, num_features_xd, 10, dropout, 'elu'),
            GraphAttBn(num_features_xd*10, output_dim, 1, dropout, 'relu'))
        self.features_drug.add_module('Encoder1', ligand_encoder)
        self.flatten = nn.Sequential(
            torch.nn.Linear(output_dim, 256), ReLU(), nn.Dropout(dropout),
            torch.nn.Linear(256, output_dim), nn.Dropout(dropout))

        # protein sequence branch (1d conv)
        self.features_target = Sequential(OrderedDict([('embed', nn.Embedding(num_features_xt + 1, embed_dim))]))
        protein_encoder = Sequential(
            nn.Conv1d(in_channels=1200, out_channels=n_filters, kernel_size=8), ReLU(),
            nn.Conv1d(in_channels=n_filters, out_channels=n_filters*2, kernel_size=8), ReLU(),
            )
        flatten_protein = Sequential(nn.Flatten(), Linear(32*2*114, output_dim))
        self.features_target.add_module('Encoder2', protein_encoder)
        self.features_target.add_module('Flatten', flatten_protein)

        # combined layers
        self.output = nn.Sequential(
            nn.Linear(2*output_dim, 1024), ReLU(), nn.Dropout(dropout),
            nn.Linear(1024, 256), ReLU(), nn.Dropout(dropout),
            nn.Linear(256, self.n_output)
        )

    def forward(self, data):

        # drug encode
        drug_x = self.features_drug(data)
        drug_x = gnn.global_max_pool(drug_x.x, drug_x.batch)
        drug_x = self.flatten(drug_x)

        # target encode
        target = data.target
        target_x = self.features_target(target)

        # concat
        concat_x = torch.cat((drug_x, target_x), 1)
        out = self.output(concat_x)
        return out