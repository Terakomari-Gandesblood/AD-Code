from model.mlp import MLPBase


class EncodingNetwork(MLPBase):
    """
        编码网络
        输入维度: input_dim
        输出维度: output_dim
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
