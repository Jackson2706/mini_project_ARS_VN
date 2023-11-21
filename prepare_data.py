import os
import xml.etree.ElementTree as ET
from glob import glob

from roboflow import Roboflow

rf = Roboflow(api_key="SHqXH9Hthjol3EotLlHd")
project = rf.workspace("ctarg").project("ars")
dataset = project.version(4).download("voc")


data_dir = "./ARS-4"
for phase in ["train", "valid", "test"]:
    image_path_list = sorted(glob(data_dir + "/" + phase + "/*.jpg"))
    xml_path_list = sorted(glob(data_dir + "/" + phase + "/*.xml"))

    for image_path, xml_path in zip(image_path_list, xml_path_list):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        if root.find("object/bndbox") is None:
            print(image_path, xml_path)
            os.remove(xml_path)
            os.remove(image_path)
