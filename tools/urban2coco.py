import os
import json
import argparse
import cv2
import shutil

class_map = {"car": "car", "person": "person", "motorcycle": "motor", "rider": "rider", "bicycle": "bike", "bus": "bus", "truck": "truck"}
CLASSES = {'bus': 0, 'bike': 1, 'car': 2, 'motor': 3, 'person': 4, 'rider': 5, 'truck': 6}

def parse_json(args):
    coco = dict()
    coco["info"] = {
        'description': 'GTAV Dataset', 
        'url': 'https://www.urbansyn.org/#tabGallery', 
        'version': '1.0', 
        'year': 2023, 
        'contributor': 'COCO Consortium', 
        'date_created': '2022/07/13'
    }

    coco["licenses"] = None

    coco["categories"] = [
        {'supercategory': 'vehicle', 'id': 0, 'name': 'bus'}, 
        {'supercategory': 'vehicle', 'id': 1, 'name': 'bike'}, 
        {'supercategory': 'vehicle', 'id': 2, 'name': 'car'}, 
        {'supercategory': 'vehicle', 'id': 3, 'name': 'motor'}, 
        {'supercategory': 'person', 'id': 4, 'name': 'person'}, 
        {'supercategory': 'person', 'id': 5, 'name': 'rider'}, 
        {'supercategory': 'vehicle', 'id': 6, 'name': 'truck'}
    ]

    img_root = os.path.join(args.root, "rgb")

    images = []
    annotations = []

    count = {'bus': 0, 'bike': 0, 'car': 0, 'motor': 0, 'person': 0, 'rider': 0, 'truck': 0}

    bnd_id = 0

    if not os.path.exists(os.path.join(args.target, "train")):
        os.makedirs(os.path.join(args.target, "train"))

    
    im_list = os.listdir(img_root)

    for i, im in enumerate(im_list):
        im_path = os.path.join(img_root, im)
        print(i, len(im_list))

        shutil.copy(im_path, os.path.join(args.target, "train/" + im))
        image = cv2.imread(im_path)
        
        h, w, _ = image.shape

        img = {
            'file_name': im, 
            'id': i, 
            'height': h, 
            'width': w
        }

        images.append(img)

        with open(os.path.join(args.root, "bbox2d", im.replace("rgb", "bbox2d").replace("png", "json")), 'r') as f:
            bboxes = json.load(f)

        for bbox in bboxes:
            category = bbox["label"]
            occlusion_percentage = bbox["occlusion_percentage"]
            if category not in class_map:
                continue

            if category == "car" or category == "person" or category == "rider":
                if category == "car":
                    if occlusion_percentage > 0.4:
                        continue
                if category == "person":
                    if occlusion_percentage > 0.1:
                        continue
                
                if category == "rider":
                    if occlusion_percentage > 30:
                        continue
            
            category_id = CLASSES[class_map[category]]
            xmin = float(bbox["bbox"]["xMin"])
            ymin = float(bbox["bbox"]["yMin"])
            xmax = float(bbox["bbox"]["xMax"])
            ymax = float(bbox["bbox"]["yMax"])

            if(xmax <= xmin or ymax <= ymin):
                continue

            count[class_map[category]] += 1

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

        #     cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
        # cv2.imwrite(f"./vis/res{i}.jpg", image)

    coco["annotations"] = annotations
    coco["images"] = images

    if not os.path.exists(os.path.join(args.target, "annotations")):
        os.makedirs(os.path.join(args.target, "annotations"))

    with open(os.path.join(args.target, "annotations/train.json"), "w") as f:
        json.dump(coco, f)

    print(count)
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert Diverse weather annotations to mmdetection format')
    parser.add_argument('--root', default="./data/UrbanSyn", help='pascal voc devkit path')
    parser.add_argument('-t', '--target', default="./data/UrbanSynCoco", help='output path')
    args = parser.parse_args()
    parse_json(args)