import torch
import torch.nn as nn
import torch.nn.functional as F


# Code made from different sources :
# https://github.com/christiancosgrove/pytorch-spectral-normalization-gan/blob/master/model_resnet.py
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

# snconv2d and SelfAttn original from: https://github.com/ajbrock/BigGAN-PyTorch
def snconv2d(eps=1e-12, **kwargs):
    return nn.utils.spectral_norm(nn.Conv2d(**kwargs), eps=eps)


class SelfAttn(nn.Module):
    def __init__(self, in_channels, eps=1e-12):
        super().__init__()
        self.in_channels = in_channels
        self.snconv1x1_theta = snconv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1,
                                        bias=False, eps=eps)
        self.snconv1x1_phi = snconv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1, bias=False,
                                      eps=eps)
        self.snconv1x1_g = snconv2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=1, bias=False,
                                    eps=eps)
        self.snconv1x1_o = snconv2d(in_channels=in_channels // 2, out_channels=in_channels, kernel_size=1, bias=False,
                                    eps=eps)
        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        _, ch, h, w = x.size()
        # Theta
        theta = self.snconv1x1_theta(x)
        theta = theta.view(-1, ch // 8, h * w)
        # Phi
        phi = self.snconv1x1_phi(x)
        phi = self.maxpool(phi)
        phi = phi.view(-1, ch // 8, h * w // 4)
        # Attn map
        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = self.softmax(attn)
        # g
        g = self.snconv1x1_g(x)
        g = self.maxpool(g)
        g = g.view(-1, ch // 2, h * w // 4)
        # Attn_g - o
        attn_g = torch.bmm(g, attn.permute(0, 2, 1))
        attn_g = attn_g.view(-1, ch // 2, h, w)
        attn_g = self.snconv1x1_o(attn_g)
        # Out
        out = x + self.gamma * attn_g
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        nn.init.orthogonal_(self.conv1.weight.data, 1.)
        self.BatchN1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.orthogonal_(self.conv2.weight.data, 1.)
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
        self.att_on = kwargs['attention']
        self.conv1 = nn.Conv2d(self.channels, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.orthogonal_(self.conv1.weight.data, 1.)
        self.BatchN1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        if self.att_on: self.attention_layer = SelfAttn(512)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.init_weights()

    # Orthogonal weight initialization (source: https://arxiv.org/abs/1312.6120)
    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight)

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
        if self.att_on: out = self.attention_layer(out)  # we use self attention if this parameter is on.
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = torch.sigmoid(self.linear(out))
        return out


class Generator_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, kernel=3, stride=1):
        super(Generator_BasicBlock, self).__init__()
        self.convTrans1 = nn.ConvTranspose2d(in_planes, planes, kernel_size=kernel, stride=stride, padding=1,
                                             bias=False)
        nn.init.orthogonal_(self.convTrans1.weight.data, 1.)
        self.BatchN1 = nn.BatchNorm2d(planes)
        self.convTrans2 = nn.ConvTranspose2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.orthogonal_(self.convTrans2.weight.data, 1.)
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
        self.att_on = kwargs['attention']
        self.factor = 8
        self.convTrans1 = nn.ConvTranspose2d(self.z_dim, self.in_planes * 8, kernel_size=4, stride=1, padding=0,
                                             bias=False)
        nn.init.orthogonal_(self.convTrans1.weight.data, 1.)
        self.BatchN1 = nn.BatchNorm2d(self.in_planes * 8)
        self.layer1 = self._make_layer(block, self.in_planes * 8, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, self.in_planes * 4, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, self.in_planes * 2, num_blocks[2], stride=2)
        if self.att_on: self.attention_layer = SelfAttn(self.in_planes)
        self.ConvTrans2 = nn.ConvTranspose2d(self.in_planes, self.channels, kernel_size=4, stride=2, padding=1)
        self.init_weights()

    # Orthogonal weight initialization (source: https://arxiv.org/abs/1312.6120)
    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.ConvTranspose2d):
                nn.init.orthogonal_(module.weight)

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
        if self.att_on: out = self.attention_layer(out)
        out = torch.tanh(self.ConvTrans2(out))
        return out
