import os
import json
import argparse
import cv2
import shutil

CLASSES = {'pedestrian': 0, 'people': 1, 'bicycle': 2, 'car': 3, 'van': 4, 'truck': 5, 'bus': 8, 'motor': 9}

class_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 8: 6, 9: 7}

CLASS_NAMES = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'bus', 'motor']

def parse_json(args):
    coco = dict()
    coco["info"] = {
        'description': 'VisDrone Dataset', 
        'url': 'https://www.urbansyn.org/#tabGallery', 
        'version': '1.0', 
        'year': 2023, 
        'contributor': 'COCO Consortium', 
        'date_created': '2022/07/13'
    }

    coco["licenses"] = None

    coco["categories"] = [
        {'supercategory': 'person', 'id': 0, 'name': 'pedestrian'}, 
        {'supercategory': 'person', 'id': 1, 'name': 'people'}, 
        {'supercategory': 'vehicle', 'id': 2, 'name': 'bicycle'}, 
        {'supercategory': 'vehicle', 'id': 3, 'name': 'car'}, 
        {'supercategory': 'vehicle', 'id': 4, 'name': 'van'}, 
        {'supercategory': 'vehicle', 'id': 5, 'name': 'truck'}, 
        {'supercategory': 'vehicle', 'id': 6, 'name': 'bus'}, 
        {'supercategory': 'vehicle', 'id': 7, 'name': 'motor'}
    ]

    images = []
    annotations = []

    txts_root = os.path.join(args.root, "labels/labels")
    img_root = os.path.join(args.root, "images/images")

    txts = os.listdir(txts_root)

    count = {}

    bnd_id = 0

    if not os.path.exists(os.path.join(args.target, "train")):
        os.makedirs(os.path.join(args.target, "train"))

    for i, txt in enumerate(txts):
        im = txt.replace('txt', 'jpg')
        im_path = os.path.join(img_root, im)
        image = cv2.imread(im_path)
        
        h, w, _ = image.shape

        img = {
            'file_name': im, 
            'id': i, 
            'height': h, 
            'width': w
        }

        shutil.copy(im_path, os.path.join(args.target, "train/" + im))

        images.append(img)

        with open(os.path.join(txts_root, txt), 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            attributes = line.strip().split(" ")
            class_id = int(attributes[0])
            category_id = class_map[class_id]

            category = CLASS_NAMES[category_id]
            if category not in CLASSES:
                continue

            center_x = int(float(attributes[1]) * w)
            center_y = int(float(attributes[2]) * h)

            tw = int(float(attributes[3]) * w)
            th = int(float(attributes[4]) * h)

            xmin = center_x - tw // 2
            xmax = center_x + tw // 2

            ymin = center_y - th // 2
            ymax = center_y + th // 2

            if(xmax <= xmin or ymax <= ymin):
                continue

            if category_id not in count:
                count[category_id] = 0
            else:
                count[category_id] += 1

            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)

            ann = {
                'area': o_width * o_height, 
                'iscrowd': 0, 
                'image_id': i, 
                'bbox':[xmin, ymin, o_width, o_height],
                'category_id': category_id, 
                'id': bnd_id, 
                'ignore': 0,
                'segmentation': []
            }

            annotations.append(ann)
            bnd_id += 1
            
    coco["annotations"] = annotations
    coco["images"] = images

    if not os.path.exists(os.path.join(args.target, "annotations")):
        os.makedirs(os.path.join(args.target, "annotations"))

    with open(os.path.join(args.target, "annotations/train.json"), "w") as f:
        json.dump(coco, f)

    print(count)
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert VisDrone annotations to mmdetection format')
    parser.add_argument('--root', default="data/visdrone_origin/train", help='pascal voc devkit path')
    parser.add_argument('-t', '--target', default="./data/VisDroneCoco", help='output path')
    args = parser.parse_args()
    parse_json(args)