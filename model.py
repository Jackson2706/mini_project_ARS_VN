import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchsummary import summary
from torchvision import transforms
from torchvision.models import resnet18


class CNN(nn.Module):
    def __init__(self, num_class: int = 10):
        super(CNN, self).__init__()
        self.model = resnet18(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.Linear(512, 128),
            nn.Linear(128, num_class),
        )

    def forward(self, X):
        out = self.model(X)
        return out

    def inference(self, img):
        self.model.eval()
        object_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        object_image = Image.fromarray(object_image)
        preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                # run normalize.py to calculate mean and std
                transforms.Normalize(
                    mean=[0.4879, 0.3781, 0.3667],
                    std=[0.2762, 0.2480, 0.2056],
                ),
            ]
        )
        image = preprocess(object_image)
        image = image.unsqueeze(0)
        model_result = self.model(image)
        result = int(torch.argmax(model_result))
        return result


if __name__ == "__main__":
    model = CNN().to("cuda")
    summary(model=model, input_size=(3, 256, 256))
