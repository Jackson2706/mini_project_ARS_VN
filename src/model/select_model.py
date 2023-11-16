from cv2 import imread
from joblib import dump
from lazypredict.Supervised import LazyClassifier
from numpy import array
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.dataset import Dataset
from src.utils.classification_preprocess import crop_object, preprocess_img


def preprocess_dataset(image_path_list, annotation_list):
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
            label_list.append(label)
    return array(image_feature_list), array(label_list)


if __name__ == "__main__":
    """
    Defining the dataset
    """
    train_dataset = Dataset(dataset_dir="License_Plate-5", phase="train")
    train_image_path_list, train_annotation_list = train_dataset.__call__()

    val_dataset = Dataset(dataset_dir="License_Plate-5", phase="test")
    val_image_path_list, val_annotation_list = val_dataset.__call__()

    """
        (Machine learning model)
    """
    """
        Preprocessing data 
    """
    # Preprocessing training data
    X_train, y_train = preprocess_dataset(
        image_path_list=train_image_path_list,
        annotation_list=train_annotation_list,
    )
    # Normalization
    scaler = StandardScaler()
    scaler.fit_transform(X_train)
    # Encode the labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)

    # Preprocessing test data
    X_val, y_val = preprocess_dataset(
        image_path_list=val_image_path_list, annotation_list=val_annotation_list
    )
    # Encode the labels
    y_val = label_encoder.transform(y_val)

    # Normalize the features
    scaler.transform(X_val)
    """
        Select model phase
    """
    # Create a list to store all of the models which are chosen to test, using lazypredict
    clf = LazyClassifier()
    clf.fit(X_train, y_train)

    """
        Validation phase
    """
