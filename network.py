import torch
import torch.nn as nn
import torch.nn.functional as torchfunc


class LinearNetwork(nn.Module):

    def __init__(self, layers_size, final_layer_function, activation_function, bias=False):
        
        self.final_layer_function = final_layer_function
        self.activation_function = activation_function

        self.bias = bias

        super().__init__()
        self.linear_layers = nn.ModuleList([nn.Linear(layers_size[i], layers_size[i + 1], bias=bias)
                                            for i in range(len(layers_size) - 1)])

    def forward(self, x):
        for i in range(len(self.linear_layers) - 1):
            x = self.linear_layers[i](x)
            x = self.activation_function(x)

        x = self.linear_layers[-1](x)
        return self.final_layer_function(x)

