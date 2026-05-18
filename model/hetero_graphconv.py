import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn.conv import HeteroConv, GraphConv
from torch_geometric.nn.norm import GraphNorm


class HeteroGraphConv(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 out_dim: int = 128,
                 num_layers: int = 3,
                 aggr: str = 'sum',
                 use_residual: bool = True,
                 dropout: float = 0.1):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.use_residual = use_residual
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.edge_types = [
            ('ego', 'to', 'vehicle'),
            ('vehicle', 'to', 'ego'),
            ('vehicle', 'to', 'vehicle'),
            ('ego', 'to', 'lane'),
            ('lane', 'to', 'ego'),
            ('ego', 'to', 'traffic_light'),
            ('traffic_light', 'to', 'ego'),
            ('ego', 'to', 'gnss'),
            ('gnss', 'to', 'ego'),
            ('ego', 'self', 'ego'),
            ('vehicle', 'self', 'vehicle'),
            ('lane', 'self', 'lane'),
            ('traffic_light', 'self', 'traffic_light'),
            ('gnss', 'self', 'gnss'),
        ]

        self.node_types = ['ego', 'vehicle', 'lane', 'traffic_light', 'gnss']

        for layer_index in range(num_layers):
            node_dim = input_dim if layer_index == 0 else hidden_dim

            conv_dict = {}
            for edge_type in self.edge_types:
                conv_dict[edge_type] = GraphConv((node_dim, node_dim), hidden_dim)
            self.layers.append(HeteroConv(conv_dict, aggr=aggr))

            self.norms.append(nn.ModuleDict({
                node_type: GraphNorm(hidden_dim) for node_type in self.node_types
            }))

        fuse_in = hidden_dim * len(self.node_types)
        self.fuse_net = nn.Sequential(
            nn.Linear(fuse_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, data: HeteroData) -> torch.Tensor:
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict

        for layer in range(self.num_layers):
            h_dict = self.layers[layer](x_dict, edge_index_dict)

            for node_type in self.node_types:
                if node_type not in h_dict:
                    h_dict[node_type] = x_dict[node_type]

            for node_type, h in h_dict.items():
                h = self.norms[layer][node_type](h)
                h = nn.functional.relu(h)
                h = self.dropout(h)

                if self.use_residual and node_type in x_dict and x_dict[node_type].shape == h.shape:
                    h = h + x_dict[node_type]
                h_dict[node_type] = h

            x_dict = h_dict

        def mean_pooling(_node_type: str):
            if _node_type in x_dict and x_dict[_node_type].size(0) > 0:
                return x_dict[_node_type].mean(dim=0, keepdim=True)
            return torch.zeros((1, self.hidden_dim), device=self._device_of_dict(x_dict))

        fused = torch.cat([
            mean_pooling('ego'),
            mean_pooling('lane'),
            mean_pooling('traffic_light'),
            mean_pooling('gnss'),
            mean_pooling('vehicle'),
        ], dim=-1)

        return self.fuse_net(fused)

    @staticmethod
    def _device_of_dict(x_dict: dict) -> torch.device:
        for v in x_dict.values():
            if isinstance(v, torch.Tensor):
                return v.device
        return torch.device('cpu')
