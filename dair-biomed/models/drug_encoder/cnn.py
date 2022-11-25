import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict

class CNN(nn.Module):
    def __init__(self, in_ch, kernels):
        super(CNN, self).__init__()
        layer_size = len(in_ch) - 1
        self.conv = nn.ModuleList(
            [nn.Conv1d(
                in_channels = in_ch[i], 
                out_channels = in_ch[i + 1], 
                kernel_size = kernels[i]
            ) for i in range(layer_size)]
        )
        self.conv = self.conv.double()

    def _get_conv_output(self, shape):
        bs = 1
        input = Variable(torch.rand(bs, *shape))
        output_feat = self._forward_features(input.double())
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def _forward_features(self, x):
        for l in self.conv:
            x = F.relu(l(x))
        x = F.adaptive_max_pool1d(x, output_size=1)
        return x
    
    def forward(self, v):
        v = self._forward_features(v.double())
        v = v.view(v.size(0), -1)
        v = self.fc1(v.float())
        return v

class DrugCNN(nn.Module):
