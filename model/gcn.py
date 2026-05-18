import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn.conv import HeteroConv, SAGEConv
from torch_geometric.nn.norm import GraphNorm


class HeteroGCN(nn.Module):
    def __init__(self,
                 input_dim: int,  # 编码网络的输出维度
                 hidden_dim: int = 128,
                 out_dim: int = 128,  # 全局向量输出维度
                 num_layers: int = 3,
                 aggr: str = 'sum',  # HeteroConv 聚合：'sum'/'mean'/'max'
                 use_residual: bool = True,
                 dropout: float = 0.1):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.use_residual = use_residual
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList()  # 卷积层
        self.norms = nn.ModuleList()  # 归一化层

        # 关系集合
        self.edge_types = [
            ('ego', 'to', 'vehicle'),
            ('vehicle', 'to', 'ego'),
            ('vehicle', 'to', 'vehicle'),
            ('ego', 'to', 'lane'),
            ('lane', 'to', 'ego'),
            ('ego', 'to', 'traffic_light'),
            ('traffic_light', 'to', 'ego'),
            # ('ego', 'to', 'imu'),
            # ('imu', 'to', 'ego'),
            ('ego', 'to', 'gnss'),
            ('gnss', 'to', 'ego'),
            ('ego', 'self', 'ego'),
            ('vehicle', 'self', 'vehicle'),
            ('lane', 'self', 'lane'),
            ('traffic_light', 'self', 'traffic_light'),
            # ('imu', 'self', 'imu'),
            ('gnss', 'self', 'gnss'),
        ]

        node_types = ['ego', 'vehicle', 'lane', 'traffic_light', 'gnss']

        for layer_index in range(num_layers):
            node_dim = input_dim if layer_index == 0 else hidden_dim

            # 构建卷积层
            conv_dict = {}
            for edge_type in self.edge_types:
                conv_dict[edge_type] = SAGEConv((node_dim, node_dim), hidden_dim)  # 所有节点编码统一维度
            self.layers.append(HeteroConv(conv_dict, aggr=aggr))

            # 构建归一化层
            self.norms.append(nn.ModuleDict({node_type: GraphNorm(hidden_dim) for node_type in node_types}))

        # 输出网络
        fuse_in = hidden_dim * len(node_types)
        self.fuse_net = nn.Sequential(
            nn.Linear(fuse_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, data: HeteroData) -> torch.Tensor:
        # 获取节点和边信息
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict

        # 消息传递
        for layer in range(self.num_layers):
            # 图卷积
            h_dict = self.layers[layer](x_dict, edge_index_dict)  # {'vehicle': (N_vehicles, hidden_dim), ...}

            for node_type, h in h_dict.items():
                h = self.norms[layer][node_type](h)  # 归一化
                h = nn.functional.relu(h)  # 激活函数
                h = self.dropout(h)  # 随机失活

                # 残差
                if self.use_residual and node_type in x_dict and x_dict[node_type].shape[1] == h.shape[1]:
                    h = h + x_dict[node_type]
                h_dict[node_type] = h

            x_dict = h_dict  # 更新输入

        # 池化
        def mean_pooling(_node_type):
            if _node_type in x_dict and x_dict[_node_type].size(0) > 0:
                return x_dict[_node_type].mean(dim=0, keepdim=True)
            else:
                return torch.zeros((1, self.hidden_dim), device=self._device_of_dict(x_dict))

        ego_h = mean_pooling('ego')
        lane_h = mean_pooling('lane')
        tl_h = mean_pooling('traffic_light')
        # imu_h = mean_pooling('imu')
        gnss_h = mean_pooling('gnss')
        vehicle_h = mean_pooling('vehicle')

        fused = torch.cat([ego_h, lane_h, tl_h, gnss_h, vehicle_h], dim=-1)  # (1, hidden_dim*5)
        global_feat = self.fuse_net(fused)  # (1, out_dim)
        return global_feat

    @staticmethod
    def _device_of_dict(x_dict: dict) -> torch.device:
        # 找到任一张量的 device
        for v in x_dict.values():
            if isinstance(v, torch.Tensor):
                return v.device
        return torch.device('cpu')
