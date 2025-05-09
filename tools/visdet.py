import os
import cv2
import random
from tqdm import tqdm
import argparse
import xml.etree.ElementTree as ET
import cv2

def xml_reader(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        name = obj.find('name').text
        if name == "motor" or name == "rider":
            bbox = obj.find('bndbox')
            obj_struct = {}
            obj_struct['name'] = name
            obj_struct['bbox'] = [float(bbox.find('xmin').text),
                                float(bbox.find('ymin').text),
                                float(bbox.find('xmax').text),
                                float(bbox.find('ymax').text)] 
            objects.append(obj_struct)
    return objects

def get_image_list(image_dir, suffix=['jpg', 'png']):
    """get all image path ends with suffix"""
    if not os.path.exists(image_dir):
        print("PATH:%s not exists" % image_dir)
        return []
    imglist = []
    for root, sdirs, files in os.walk(image_dir):
        if not files:
            continue
        for filename in files:
            filepath = os.path.join(root, filename)
            if filename.split('.')[-1] in suffix:
                imglist.append(filepath)
    return imglist

if __name__ == "__main__":
    root = "./data/Diverse_weather/daytime_clear/VOC2007"

    annotations = os.listdir(os.path.join(root, "Annotations"))
    for i, ann in enumerate(annotations):
        ann_path = os.path.join(root, "Annotations", ann)
        objs = xml_reader(ann_path)

        if len(objs) > 0:
            image = cv2.imread(ann_path.replace('Annotations', 'JPEGImages').replace("xml", "jpg"))
            for obj in objs:
                if obj['name'] == 'rider':
                    name = obj['name']
                    box = obj['bbox']
                    p1 = (int(box[0]), int(box[1]))
                    p2 = (int(box[2]), int(box[3]))
                    # p3 = (max(box[0], 15), max(box[1], 15))
                    cv2.rectangle(image, p1, p2, (0, 0, 255), 2)
                    # cv2.putText(img, name, p3, cv2.FONT_ITALIC, 1, (0, 255, 0), 2)
                else:
                    name = obj['name']
                    box = obj['bbox']
                    p1 = (int(box[0]), int(box[1]))
                    p2 = (int(box[2]), int(box[3]))
                    # p3 = (max(box[0], 15), max(box[1], 15))
                    cv2.rectangle(image, p1, p2, (255, 0, 0), 2)
            cv2.imwrite(f"./vis/res{i}.jpg", image)
