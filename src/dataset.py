import xml.etree.ElementTree as ET
from glob import glob
from cv2 import imread

class Dataset:
    def __init__(self, dataset_dir = "License_Plate-5", phase = "train"):
        self.dataset_dir = dataset_dir
        self.phase = phase
        self.images_list = []
        self.annotations_list = []


    '''
        Create a dataset to train a model as well as validate the training procedure
        returns a list of images and annotations
    '''
    def __call__(self):
        self.images_list = sorted(glob(self.dataset_dir + "/"+self.phase + "/*.jpg"))
        xml_path_list = sorted(glob(self.dataset_dir + "/"+self.phase + "/*.xml"))
        for xml_path in  xml_path_list:
            # parse the xml file
            tree  = ET.parse(xml_path)
            root = tree.getroot()
            
            # store the location of objects
            objects_list = []
            for object_ in root.findall("object"):
                label = object_.find("name").text
                xmin = int(object_.find("bndbox/xmin").text)
                ymin = int(object_.find("bndbox/ymin").text)
                xmax = int(object_.find("bndbox/xmax").text)
                ymax = int(object_.find("bndbox/ymax").text)
                objects_list.append([xmin, ymin, xmax, ymax, label])
            self.annotations_list.append(objects_list)
        return self.images_list, self.annotations_list
    

if __name__ == "__main__":
    from cv2 import rectangle, imwrite
    dataset = Dataset()
    image_list, annotation_list = dataset.__call__()
    idx = 155
    image, annotations = image_list[idx], annotation_list[idx]
    for annotation in annotations:
        [x1,y1,x2,y2,labels] = annotation

        rectangle(image, (x1,y1), (x2,y2), (0,255,0),2)
    imwrite("image.jpg",image)
