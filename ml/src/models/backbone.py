import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return torch.relu(out)


class CoughResNet(nn.Module):
    def __init__(self, in_channels=1, base_filters=32, layers=[2, 2, 2, 2]):
        super().__init__()
        self.in_planes = base_filters

        self.conv1 = nn.Conv2d(in_channels, base_filters, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_filters)

        self.layer1 = self._make_layer(base_filters, layers[0], stride=1)
        self.layer2 = self._make_layer(base_filters * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(base_filters * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(base_filters * 8, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_features = base_filters * 8

    def _make_layer(self, planes, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, 1, n_mels, time)
        x = torch.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feature_map = self.layer4(x)  # For Grad-CAM

        embedding = self.avgpool(feature_map).flatten(1)  # (B, out_features)
        return embedding, feature_map


class TierHead(nn.Module):
    def __init__(self, in_features, num_tiers=3, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, num_tiers)
        )

    def forward(self, embedding):
        # embedding: (B, in_features)
        return self.net(embedding)  # (B, num_tiers)


if __name__ == '__main__':
    model = CoughResNet()
    x = torch.randn(1, 1, 64, 157)
    emb, feat = model(x)
    print(f'Embedding: {emb.shape}, Feature map: {feat.shape}')

    head = TierHead(model.out_features)
    logits = head(emb)
    print(f'Logits: {logits.shape}')