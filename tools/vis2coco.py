import os
import json
import argparse
import cv2
import shutil

CLASSES = {'pedestrian': 1, 'people': 2, 'bicycle': 3, 'car': 4, 'van': 5, 'truck': 6, 'bus': 9, 'motor': 10}

class_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 9: 6, 10: 7}

CLASS_NAMES = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning tricycle', 'bus', 'motor']

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
        {'supercategory': 'vehicle', 'id':7, 'name': 'motor'}
    ]

    images = []
    annotations = []

    txts_root = os.path.join(args.root, "labels")
    img_root = os.path.join(args.root, "images")

    txts = os.listdir(txts_root)
    bnd_id = 0

    count = {}

    if not os.path.exists(os.path.join(args.target, "origin_train")):
        os.makedirs(os.path.join(args.target, "origin_train"))

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

        shutil.copy(im_path, os.path.join(args.target, "origin_train/" + im))

        images.append(img)


        with open(os.path.join(txts_root, txt), 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            attributes = line.strip().split(",")
            class_id = int(attributes[5])

            if class_id not in class_map:
                continue
            
            category = CLASS_NAMES[class_map[class_id]]
            
            if category not in CLASSES or int(attributes[4]) == 0 or int(attributes[6]) != 0 or int(attributes[7]) != 0:
                continue

            category_id = class_map[class_id]
            xmin = int(float(attributes[0]))
            ymin = int(float(attributes[1]))
            xmax = xmin + int(float(attributes[2]))
            ymax = ymin + int(float(attributes[3]))

            # cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            # cv2.putText(image, category, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        

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
        # cv2.imwrite(f"./vis/res{i}.jpg", image)
    coco["annotations"] = annotations
    coco["images"] = images

    if not os.path.exists(os.path.join(args.target, "annotations")):
        os.makedirs(os.path.join(args.target, "annotations"))

    with open(os.path.join(args.target, "annotations/origin_train.json"), "w") as f:
        json.dump(coco, f)

    print(count)
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert VisDrone annotations to mmdetection format')
    parser.add_argument('--root', default="data/visdrone_origin/origin_train", help='pascal voc devkit path')
    parser.add_argument('-t', '--target', default="./data/VisDroneCoco/All", help='output path')
    args = parser.parse_args()
    parse_json(args)