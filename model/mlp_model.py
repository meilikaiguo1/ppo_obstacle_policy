import torch
import torch.nn as nn
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    if hasattr(layer, "weight") and layer.weight is not None:
        torch.nn.init.orthogonal_(layer.weight, std)
    if hasattr(layer, "bias") and layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# def mlp(sizes, activation, output_activation=nn.Identity, dropout=0.05):
#     """
#     :param dropout: dropout比例
#     :param sizes: 列表，长度表示隐藏层层数，元素表示该层神经元数
#     :param activation: 隐藏层激活函数
#     :param output_activation: 输出层激活函数
#     :return: nn.Sequential
#     """
#     layers = []
#     for j in range(len(sizes) - 1):
#         act = activation if j < len(sizes) - 2 else output_activation
#         if j < len(sizes) - 2:
#             layers += [nn.Linear(sizes[j], sizes[j + 1]), nn.Dropout(p=dropout), act()]
#         else:
#             if act is nn.Softmax:
#                 layers += [nn.Linear(sizes[j], sizes[j + 1]), act(dim=0)]
#             else:
#                 layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
#     return nn.Sequential(*layers)



def mlp(sizes, activation, output_activation=nn.Identity):
    """
    :param dropout: dropout比例
    :param sizes: 列表，长度表示隐藏层层数，元素表示该层神经元数
    :param activation: 隐藏层激活函数
    :param output_activation: 输出层激活函数
    :return: nn.Sequential
    """
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        if j < len(sizes) - 2:
            layers += [layer_init(nn.Linear(sizes[j], sizes[j + 1])), act()]
        else:
            if act is nn.Softmax:
                layers += [layer_init(nn.Linear(sizes[j], sizes[j + 1])), act(dim=0)]
            else:
                layers += [layer_init(nn.Linear(sizes[j], sizes[j + 1])), act()]
    return nn.Sequential(*layers)


