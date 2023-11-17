import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_feature: int = 324, num_class: int = 10):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(in_features=in_feature, out_features=256)
        self.layer2 = nn.Linear(in_features=256, out_features=64)
        self.layer3 = nn.Linear(in_features=64, out_features=16)
        self.layer4 = nn.Linear(in_features=16, out_features=num_class)

    def forward(self, X):
        out = self.layer1(X)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out
