import torch.nn as nn
from torchsummary import summary


class CNN(nn.Module):
    def __init__(self, num_class: int = 10):
        super(CNN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=8,
                padding="same",
                stride=1,
                kernel_size=(3, 3),
            ),
            nn.SiLU(),
            nn.BatchNorm2d(num_features=8),
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                padding="same",
                stride=1,
                kernel_size=(3, 3),
            ),
            nn.SiLU(),
            nn.BatchNorm2d(num_features=16),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                padding="same",
                stride=1,
                kernel_size=(3, 3),
            ),
            nn.SiLU(),
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                padding="valid",
                stride=1,
                kernel_size=(3, 3),
            ),
            nn.SiLU(),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                padding="same",
                stride=1,
                kernel_size=(3, 3),
            ),
            nn.SiLU(),
            nn.BatchNorm2d(num_features=128),
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                padding="same",
                stride=1,
                kernel_size=(3, 3),
            ),
            nn.SiLU(),
            nn.BatchNorm2d(num_features=256),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(),
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                padding="same",
                stride=1,
                kernel_size=(3, 3),
            ),
            nn.SiLU(),
            nn.BatchNorm2d(num_features=512),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(
                in_channels=512,
                out_channels=1024,
                padding="same",
                stride=1,
                kernel_size=(3, 3),
            ),
            nn.SiLU(),
            nn.BatchNorm2d(num_features=1024),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(),
        )
        self.FC = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=7 * 7 * 1024, out_features=1024),
            nn.SiLU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.SiLU(),
            nn.Linear(512, num_class),
        )

    def forward(self, X):
        out = self.block1(X)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.FC(out)
        return out


if __name__ == "__main__":
    model = CNN().to("cuda")
    summary(model=model, input_size=(3, 256, 256))
