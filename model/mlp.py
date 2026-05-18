import math
import torch
import torch.nn as nn
from typing import Optional


class MLPBase(nn.Module):
    def __init__(
            self,
            num_layers: int,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            activation: Optional[nn.Module] = None,
            use_layer_norm: bool = True,
            dropout: float = 0.0,
            use_residual: bool = False,
            final_init_std: float = 1e-3,
    ):
        super().__init__()
        assert num_layers >= 1
        self.num_layers = num_layers
        self.use_residual = use_residual and (num_layers > 1)
        self.activation = activation or nn.ReLU()
        self.dropout = dropout
        self.final_init_std = final_init_std

        if num_layers == 1:
            self.layers = nn.ModuleList([nn.Linear(input_dim, output_dim)])
            self.norms = nn.ModuleList()  # empty
        else:
            layers = [nn.Linear(input_dim, hidden_dim)]
            # input -> hidden
            # hidden -> hidden
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            # last hidden -> output
            layers.append(nn.Linear(hidden_dim, output_dim))
            self.layers = nn.ModuleList(layers)

            # norms
            norms = []
            for i in range(num_layers - 1):
                norms.append(nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity())
            self.norms = nn.ModuleList(norms)

        # dropout 层
        self._dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # 初始化权重
        self.reset_parameters()

    def reset_parameters(self):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                # 对隐藏层使用 orthogonal 初始化，gain 根据激活函数调整
                if i < len(self.layers) - 1:
                    # 用于 ReLU 的 gain = sqrt(2)
                    gain = math.sqrt(2.0) if isinstance(self.activation, (nn.ReLU,)) else 1.0
                    nn.init.orthogonal_(layer.weight, gain=gain)
                    nn.init.zeros_(layer.bias)
                else:
                    # 最后一层
                    nn.init.normal_(layer.weight, mean=0.0, std=self.final_init_std)
                    nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_layers == 1:
            return self.layers[0](x)

        for i in range(self.num_layers - 1):
            x_in = x  # 残差判断
            x = self.layers[i](x_in)
            # norms 对应的是隐藏层维度（hidden_dim）
            x = self.norms[i](x)
            x = self.activation(x)
            x = self._dropout(x)

            # 残差
            if self.use_residual and (x_in.shape[-1] == x.shape[-1]):
                x = x + x_in

        # 最后一层
        x = self.layers[-1](x)
        return x


class EncodingNetwork(MLPBase):
    def __init__(self, *args, normalize_output: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.normalize_output = normalize_output
        self._out_norm = nn.LayerNorm(kwargs["output_dim"]) if normalize_output else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        return self._out_norm(x)


class CriticNetwork(MLPBase):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim=1):
        super().__init__(num_layers, input_dim, hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = super().forward(x)
        return value.squeeze(-1)  # [batch]


def main():
    torch.manual_seed(42)
    batch_size = 8
    input_dim = 64

    # 测试编码器
    encoder = EncodingNetwork(
        num_layers=3,
        input_dim=input_dim,
        hidden_dim=128,
        output_dim=32,
        use_layernorm=True,
        use_residual=True,
        normalize_output=True,
    )
    x = torch.randn(batch_size, input_dim)
    encoded = encoder(x)
    print("=== EncodingNetwork Test ===")
    print("Input shape:", x.shape)
    print("Encoded output shape:", encoded.shape)
    print("First output sample:", encoded[0, :5])
    print("Parameter count:", sum(p.numel() for p in encoder.parameters()))
    print()

    # 测试评论家网络
    critic = CriticNetwork(num_layers=3, input_dim=32, hidden_dim=128)
    value = critic(encoded)
    print("=== CriticNetwork Test ===")
    print("Input shape:", encoded.shape)
    print("Value output shape:", value.shape)
    print("Value sample:", value[:5])
    print("Parameter count:", sum(p.numel() for p in critic.parameters()))
    print()

    # 梯度检查
    loss = value.mean()
    loss.backward()
    print("Gradients computed (example layer):", encoder.layers[0].weight.grad.norm().item())


if __name__ == "__main__":
    main()
