import torch
import torch.nn as nn


class Squeeze_and_Excitation_block(nn.Module):
    def __init__(self, in_channels, reduction_ratio=2):
        super(Squeeze_and_Excitation_block, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(in_channels, in_channels//reduction_ratio)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channels//reduction_ratio, in_channels)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        batch_size, num_channels, _ = x.size()
        residual = x

        out = self.avgpool(x)
        
        out = out.view(batch_size, num_channels)

        out = self.fc1(out)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.sigmoid(out)

        out = out.view(batch_size, num_channels, 1)

        out = residual * out.expand_as(x)
        return out
    

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(residual)
        out = self.relu(out)

        return out

class ResNet18_SE(nn.Module):
    def __init__(self, in_channels, num_classes=2):
        super(ResNet18_SE, self).__init__()
        self.in_channels = in_channels

        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=2)

        self.in_channels = 64

        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2)
        self.layer3 = self._make_layer(256, 2)
        self.layer4 = self._make_layer(512, 2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, blocks):
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels))
        self.in_channels = out_channels
        for _ in range(0, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))
        layers.append(nn.MaxPool1d(kernel_size=2))
        layers.append(Squeeze_and_Excitation_block(self.in_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

