import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, layer_widths=[100,100,2], activate_final=False, activation_fn=F.tanh):
        super(MLP, self).__init__()
        layers = []
        prev_width = input_dim
        for layer_width in layer_widths:
            layers.append(nn.Linear(prev_width, layer_width))
            prev_width = layer_width
        self.input_dim = input_dim
        self.layer_widths = layer_widths
        self.layers = nn.ModuleList(layers)
        self.activate_final = activate_final
        self.activation_fn = activation_fn

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation_fn(layer(x))
        x = self.layers[-1](x)
        if self.activate_final:
            x = self.activation_fn(x)
        return x


class MLPScoreNetwork(nn.Module):
    def __init__(self, input_dim, layer_widths=[100, 100, 2], activate_final=False, activation_fn=F.tanh):
        super().__init__()
        self.locals = [input_dim, layer_widths, activate_final, activation_fn]
        self.net = MLP(input_dim, layer_widths=layer_widths, activate_final=activate_final, activation_fn=activation_fn)

    def forward(self, x, y, t):
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)
