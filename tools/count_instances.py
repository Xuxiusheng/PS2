import json_lines

jsonl_file = "/home/xuxiusheng/project/Open-GroundingDino/data/UrbanSynCoco/annotations/train.jsonl"

CLASSES = {'bus': 0, 'bike': 0, 'car': 0, 'motor': 0, 'person': 0, 'rider': 0, 'truck': 0}
SMALL_CLASSES = {'bus': 0, 'bike': 0, 'car': 0, 'motor': 0, 'person': 0, 'rider': 0, 'truck': 0}

with open(jsonl_file, 'rb') as f:
    for obj in json_lines.reader(f):
        detection = obj['detection']['instances']
        for it in detection:
            CLASSES[it['category']] += 1
            xyxy = it['bbox']
            w = xyxy[2] - xyxy[0]
            h = xyxy[3] - xyxy[1]
        
            if w < 32 and h < 32:
                SMALL_CLASSES[it['category']] += 1

print(CLASSES, SMALL_CLASSES)