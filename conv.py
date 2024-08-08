import torch
import torch.nn as nn

class swish(nn.Module):
    def __init__(self):
        super(swish, self).__init__()

    def forward(self, x):
        return (x * torch.exp(x)) / (torch.exp(x) + 1)

class DepthwiseConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(DepthwiseConv1d, self).__init__()
        assert out_channels % in_channels == 0

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=bias
        )

    def forward(self, inputs):
        output = self.conv(inputs)
        return output

class PointwiseConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, padding=0, bias=False):
        super(PointwiseConv1d, self).__init__()

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding,
            bias=bias
        )

    def forward(self, inputs):
        output = self.conv(inputs)
        return output

# Conv Module

class ConvModule(nn.Module):


    def __init__(self, in_channels, kernel_size=31, expansion_factor=2, dropout_p=.1):
        super(ConformerConvModule, self).__init__()
        assert (kernel_size -1) % 2 == 0 
        assert expansion_factor == 2
        self.layernorm = nn.LayerNorm(in_channels)
        self.depthwiseConv1d = DepthwiseConv1d(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size -1) // 2)
        self.batchnorm = nn.BatchNorm1d(in_channels)
        self.Swish = swish()
        self.pointwiseConv1d_b = PointwiseConv1d(in_channels, in_channels, stride=1, padding=0, bias=True)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, inputs):

        x = self.layernorm(inputs)
        x = x.transpose(1, 2)
        x = self.depthwiseConv1d(x)
        x = self.batchnorm(x)
        x = self.Swish(x)
        x = self.pointwiseConv1d_b(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        x = x + inputs

        return x

































