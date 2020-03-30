import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        self.BatchN1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)
        self.BatchN2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.BatchN1(self.conv1(x)))
        out = self.BatchN2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_Discriminator(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1, **kwargs):
        super(ResNet_Discriminator, self).__init__()
        self.in_planes = kwargs['in_planes']
        self.channels = kwargs['channels']

        self.conv1 = nn.Conv2d(self.channels, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        self.BatchN1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.BatchN1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = torch.sigmoid(self.linear(out)) #todo this sigmoid might be wrong check !
        return out


class Generator_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, kernel=3, stride=1):
        super(Generator_BasicBlock, self).__init__()
        self.convTrans1 = nn.ConvTranspose2d(in_planes, planes, kernel_size=kernel, stride=stride, padding=1,
                                             bias=False)
        nn.init.xavier_uniform_(self.convTrans1.weight.data, 1.)
        self.BatchN1 = nn.BatchNorm2d(planes)
        self.convTrans2 = nn.ConvTranspose2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.xavier_uniform_(self.convTrans2.weight.data, 1.)
        self.BatchN2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(in_planes, self.expansion * planes, kernel_size=2, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
        if in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(in_planes, self.expansion * planes, kernel_size=3, stride=stride, padding=1,
                                   bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.BatchN1(self.convTrans1(x)))
        out = self.BatchN2(self.convTrans2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_Generator(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1, **kwargs):
        super(ResNet_Generator, self).__init__()
        self.z_dim = kwargs['z_dim']
        self.in_planes = kwargs['in_planes']
        self.channels = kwargs['channels']
        self.factor = 8
        self.convTrans1 = nn.ConvTranspose2d(self.z_dim, self.in_planes * 8, kernel_size=4, stride=1, padding=0,
                                             bias=False)
        nn.init.xavier_uniform_(self.convTrans1.weight.data, 1.)
        self.BatchN1 = nn.BatchNorm2d(self.in_planes * 8)
        self.layer1 = self._make_layer(block, self.in_planes * 8, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, self.in_planes * 4, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, self.in_planes * 2, num_blocks[2], stride=2)
        self.ConvTrans2 = nn.ConvTranspose2d(self.in_planes, self.channels, kernel_size=4, stride=2, padding=1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        kernels = [4] + [3] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            if i == (num_blocks - 1):
                layers.append(block(self.in_planes * self.factor, planes // 2, kernels[i], strides[i]))
            else:
                layers.append(block(self.in_planes * self.factor, planes, kernels[i], strides[i]))
        self.factor = self.factor // 2
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.BatchN1(self.convTrans1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = torch.tanh(self.ConvTrans2(out))
        return out