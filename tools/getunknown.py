from pycocotools.coco import COCO
import os
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw
 
# 需要设置的路径
savepath="/path/to/generate/COCO/" 
img_dir=savepath+'images/'
annFile='/root/autodl-tmp/coco/labels/annotations/instances_val2017.json'
datasets_list='val2017'

classes_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
              'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
              'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
              'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
              'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
              'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
              'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
              'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
              'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
              'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
              'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
              'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
              'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
              'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

coco = COCO(annFile)
 
def id2name(coco):
    classes=dict()
    for cls in coco.dataset['categories']:
        classes[cls['id']]=cls['name']
    return classes

#获取COCO数据集中的所有类别
classes = id2name(coco)
print(classes)
#[1, 2, 3, 4, 6, 8]
classes_ids = coco.getCatIds(catNms=classes_names)
print(classes_ids)
image_num = set()
for cls in classes_names:
    #获取该类的id
    cls_id=coco.getCatIds(catNms=[cls])
    img_ids=coco.getImgIds(catIds=cls_id)
    if cls == "bench":
        print("cls:", cls)
        print("img:", img_ids)
        break
    print(cls,len(img_ids),img_ids)
    image_num.update(set(img_ids))
print("unknown image nums:",len(image_num))

