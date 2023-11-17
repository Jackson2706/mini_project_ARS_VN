import torch.nn as nn
from torchsummary import summary


class CNN(nn.Module):
    def __init__(self, num_class: int = 10):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=(3, 3),
                stride=1,
                padding="same",
            ),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=16),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(3, 3),
                stride=1,
                padding="same",
            ),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=32),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=8,
                kernel_size=(3, 3),
                stride=1,
                padding="same",
            ),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=8),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

        self.FC = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=128),
            nn.Linear(in_features=128, out_features=32),
            nn.Linear(in_features=32, out_features=num_class),
        )

    def forward(self, X):
        out = self.conv1(X)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.FC(out)
        return out


if __name__ == "__main__":
    model = CNN().to("cuda")
    summary(model=model, input_size=(3, 64, 64))
