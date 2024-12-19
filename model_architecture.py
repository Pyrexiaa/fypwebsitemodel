import torch.nn.functional as F
from torch import nn


class FNNClassifierTri3(nn.Module):
    """Docs for model architecture.

    : param input_size : number of input features.
    : param dropout_rate: dropout probablity
    : param layer_output_size: number of output features each layer
    """

    def __init__(self, input_size : int, dropout_rate : float, layer_output_size: int):
        super().__init__()
        self.input_size = input_size
        self.dropout_rate = dropout_rate
        self.layer_output_size = layer_output_size

        self.layer1 = nn.Linear(self.input_size, self.layer_output_size)
        self.layer2 = nn.Linear(self.layer_output_size, self.layer_output_size*2)
        self.layer3 = nn.Linear(self.layer_output_size*2, self.layer_output_size)
        self.layer4 = nn.Linear(self.layer_output_size, 1)

        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward_layer(self, layer, x):
      x = layer(x)
      x = F.mish(x)
      return self.dropout(x)

    def get_x_after_first_layer(self, x):
      return self.forward_layer(self.layer1, x)

    def get_x_after_second_layer(self, x):
      x_after_first_layer = self.get_x_after_first_layer(x)
      return self.forward_layer(self.layer2, x_after_first_layer)

    def get_x_after_third_layer(self, x):
      x_after_second_layer = self.get_x_after_second_layer(x)
      return self.forward_layer(self.layer3, x_after_second_layer)

    def forward(self, x):
      x_after_third_layer = self.get_x_after_third_layer(x)
      return self.layer4(x_after_third_layer)
