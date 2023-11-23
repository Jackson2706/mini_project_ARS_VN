import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

encode_label = {
    "dilmah": 0,
    "g7": 1,
    "jack-jill": 2,
    "karo": 3,
    "nestea_atiso": 4,
    "nestea_chanh": 5,
    "nestea_hoaqua": 6,
    "orion": 7,
    "tipo": 8,
    "y40": 9,
}


class MyDataset(Dataset):
    def __init__(self, dataset_dir, phase):
        self.image_path_list = []
        self.annotation_list = []
        data = os.path.join(dataset_dir, phase)
        for label in os.listdir(data):
            image_folder = os.path.join(data, label)
            for image in os.listdir(image_folder):
                image_path = os.path.join(image_folder, image)
                self.image_path_list.append(image_path)
                self.annotation_list.append(label)

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, index):
        image_path = self.image_path_list[index]
        label = self.annotation_list[index]
        image = Image.open(image_path)
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
        input_tensor = preprocess(image)

        label = encode_label[label]
        label = torch.tensor(label)
        return input_tensor, label
