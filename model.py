import torch.nn as nn


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                                   bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ResidualBlock(nn.Module):

    def __init__(self, in_channeld, out_channels):
        super(ResidualBlock, self).__init__()

        self.residual_conv = nn.Conv2d(in_channels=in_channeld, out_channels=out_channels, kernel_size=1, stride=2,
                                       bias=False)
        self.residual_bn = nn.BatchNorm2d(out_channels, momentum=0.99, eps=1e-3)

        self.sepConv1 = SeparableConv2d(in_channels=in_channeld, out_channels=out_channels, kernel_size=3, bias=False,
                                        padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=0.99, eps=1e-3)
        self.relu = nn.ReLU()

        self.sepConv2 = SeparableConv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, bias=False,
                                        padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.99, eps=1e-3)
        self.maxp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        res = self.residual_conv(x)
        res = self.residual_bn(res)
        x = self.sepConv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.sepConv2(x)
        x = self.bn2(x)
        x = self.maxp(x)
        return res + x


class Model(nn.Module):

    def __init__(self, num_classes):
        super(Model, self).__init__()

        # use for Xception model
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(8, momentum=0.99, eps=1e-3)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(8, momentum=0.99, eps=1e-3)
        self.relu2 = nn.ReLU()
        
        self.resblock1 = ResidualBlock(in_channeld=8, out_channels=16)
        self.resblock2 = ResidualBlock(in_channeld=16, out_channels=32)
        self.resblock3 = ResidualBlock(in_channeld=32, out_channels=64)
        self.resblock4 = ResidualBlock(in_channeld=64, out_channels=128)
        
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=3, padding=1, bias=False)
        self.adapt_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, input):

        # use for Xception model
        input = self.conv1(input)
        input = self.batch_norm1(input)
        input = self.relu1(input)
        input = self.conv2(input)
        input = self.batch_norm2(input)
        input = self.relu2(input)
        input = self.resblock1(input)
        input = self.resblock2(input)
        input = self.resblock3(input)
        input = self.resblock4(input)
        input = self.conv3(input)
        input = self.adapt_pool(input)
        input = input.view(input.size(0), -1)
        return input
        