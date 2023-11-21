import xml.etree.ElementTree as ET
from glob import glob

import torch
from cv2 import imread, resize
from numpy import array
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from utils.classification_preprocess import crop_object, preprocess_img

encode_label = {
    "dilmah": 0,
    "g7": 1,
    "jack_jill": 2,
    "karo": 3,
    "nestea_atiso": 4,
    "nestea_chanh": 5,
    "nestea_hoaqua": 6,
    "orion": 7,
    "tipo": 8,
    "y40": 9,
}


class MyDataset(Dataset):
    def __init__(self, dataset_dir="ARS-4", phase="train", hog=True):
        self.dataset_dir = dataset_dir
        self.phase = phase
        self.hog = hog
        self.images_list = []
        self.annotations_list = []
        _, _ = self.__call__()

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):
        image_path = self.images_list[index]
        [[x1, y1, x2, y2, label]] = self.annotations_list[index]
        image_ = imread(image_path)
        object_image = crop_object(image_, [x1, y1, x2, y2])
        if self.hog:
            image = preprocess_img(object_image)
            image = torch.FloatTensor(image)
        else:
            image = resize(object_image, (256, 256))
            image = torch.from_numpy(image[:, :, (2, 1, 0)]).permute(2, 0, 1)
            image = image.to(torch.float32)
        label = encode_label[label]
        label = torch.tensor(label)
        return image, label

    def _preprocess_dataset(self, image_path_list, annotation_list):
        """
        Preprocessing dataset before training

        @param:
            image_path_list: List[str], list of the path of image
            annotation_list: List[str], list of label ordered by image index

        @return:
            an array of a list of label, an array contains a list of image represenations via HoG
        """

        image_feature_list = []
        label_list = []
        for image_path, annotation in zip(image_path_list, annotation_list):
            for [x1, y1, x2, y2, label] in annotation:
                image = imread(image_path)
                object_img = crop_object(image, [x1, y1, x2, y2])
                hog_object_image = preprocess_img(object_img)
                image_feature_list.append(hog_object_image)
                label_list.append(encode_label[label])
        return array(image_feature_list), array(label_list)

    def __call__(self, tranforms=True):
        """
        Create a dataset to train a model as well as validate the training procedure
        returns a list of images and annotations
        """
        self.images_list = sorted(
            glob(self.dataset_dir + "/" + self.phase + "/*.jpg")
        )
        xml_path_list = sorted(
            glob(self.dataset_dir + "/" + self.phase + "/*.xml")
        )
        for xml_path in xml_path_list:
            # parse the xml file
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # store the location of objects
            objects_list = list()
            for object_ in root.findall("object"):
                label = object_.find("name").text
                xmin = int(object_.find("bndbox/xmin").text)
                ymin = int(object_.find("bndbox/ymin").text)
                xmax = int(object_.find("bndbox/xmax").text)
                ymax = int(object_.find("bndbox/ymax").text)
                objects_list.append([xmin, ymin, xmax, ymax, label])
            self.annotations_list.append(objects_list)
        if tranforms:
            tr_images_list, tr_annotations_list = self._preprocess_dataset(
                self.images_list, self.annotations_list
            )
        return tr_images_list, tr_annotations_list


if __name__ == "__main__":
    from cv2 import imwrite, rectangle

    dataset = Dataset()
    image_list, annotation_list = dataset.__call__()
    idx = 155
    image, annotations = image_list[idx], annotation_list[idx]
    for annotation in annotations:
        [x1, y1, x2, y2, labels] = annotation

        rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    imwrite("image.jpg", image)
