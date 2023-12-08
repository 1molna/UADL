# Convert datasets into open-set/closed-set forms.

import argparse
import tqdm

import matplotlib
import matplotlib.pyplot as plt

import os
import shutil
from mmdet.datasets import build_dataset
import numpy as np
import json
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from base_dirs import *

def parse_args():
	parser = argparse.ArgumentParser(description='Test with Distance')
	parser.add_argument('--dataset', default = 'kitti', help='kitti, idd, bdd100k, voc or coco')
	args = parser.parse_args()
	return args

args = parse_args()

print(f'Converting {args.dataset} to a closed-set form.')

BASE_DATA_FOLDER = "/root/autodl-tmp/VOC"

img_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(608, 608),
        flip=False,
        transforms=[
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
###################### Load in datasets ########################
if args.dataset == 'voc':

    vocData2007 = dict(samples_per_gpu = 1, workers_per_gpu = 4,
                train = dict(type = 'VOCDataset',
                    ann_file=  BASE_DATA_FOLDER+'/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt',
                    img_prefix= BASE_DATA_FOLDER+'/VOCdevkit/VOC2007/',
                    pipeline=img_pipeline),
                    test = dict(type = 'VOCDataset',
                    ann_file=  BASE_DATA_FOLDER+'/VOCdevkit/VOC2007/ImageSets/Main/test.txt',
                    img_prefix= BASE_DATA_FOLDER+'/VOCdevkit/VOC2007/',
                    pipeline=img_pipeline))

    vocData2012 = dict(samples_per_gpu = 1, workers_per_gpu = 4,
                train = dict(type = 'VOCDataset',
                    ann_file=  BASE_DATA_FOLDER+'/VOCdevkit/VOC2012/ImageSets/Main/trainval.txt',
                    img_prefix= BASE_DATA_FOLDER+'/VOCdevkit/VOC2012/',
                    pipeline=img_pipeline))

    
    print('Building datasets')
    vocDataset2007Train = build_dataset(vocData2007['train'])
    vocDataset2012Train = build_dataset(vocData2012['train'])
    vocDataset2007Test = build_dataset(vocData2007['test'])

    #for each dataset, collect classes in each image and their filenames
    imClasses = {'2007Train': [], '2012Train': [], '2007Test': []}
    fileNames = {'2007Train': [], '2012Train': [], '2007Test': []}

    dNames = ['2007Train', '2012Train', '2007Test']
    for dIdx, dataset in enumerate([vocDataset2007Train, vocDataset2012Train, vocDataset2007Test]):
        for imIdx in range(len(dataset)):
            imInfo = dataset.get_ann_info(imIdx)
            clsesPresent = list(imInfo['labels']) + list(imInfo['labels_ignore'])
            imClasses[dNames[dIdx]] += [clsesPresent]
            fileNames[dNames[dIdx]] += [dataset.data_infos[imIdx]['filename']]


    # For VOC, the first 10 classes (0-9) are 'known' classes, and the rest are 'unknown'
    cutoffCls = 9

    totalTrainImgs = len(vocDataset2007Train) + len(vocDataset2012Train)
    includedTrainImgs = 0
    totalTrainInstances = [0 for i in range(20)]
    includedTrainInstances = [0 for i in range(20)]
    filesIncluded = {'2007Train': [], '2012Train': [], '2007Test':[]}

    for dIdx, dName in enumerate(dNames):
        for imIdx, imCls in enumerate(imClasses[dName]):
            #statistics of original instance distribution in training data
            if 'Train' in dName:
                for cl in imCls:
                    totalTrainInstances[cl] += 1

            mask = np.asarray(imCls) > cutoffCls

            #if the image has any 'unknown' classes, it is not included in the new training dataset
            if np.sum(mask) != 0:
                continue

            #otherwise it is included in the new training dataset
            filesIncluded[dName] += [fileNames[dName][imIdx]]

            #statistics of new instance distribution in training data
            if 'Train' in dName:
                for cl in imCls:
                    includedTrainInstances[cl] += 1
                includedTrainImgs += 1

    #let's check the data balance in our new training dataset
    plt.figure()
    plt.plot(includedTrainInstances, label = 'New Training dataset')
    plt.plot(totalTrainInstances, label = 'Original Training dataset')
    plt.ylabel('Number of instances')
    plt.xlabel('Class ID')
    plt.legend()
    plt.title('Number of class instances between original and new training dataset')
    plt.show()

    #data balance as a percent of original training dataset?
    percentInstances = np.array(includedTrainInstances)/np.array(totalTrainInstances)
    plt.figure()
    plt.plot(percentInstances)
    plt.ylabel('% class instances retained')
    plt.xlabel('Class ID')
    plt.title('Percent of class instances retained in new training dataset')
    plt.show()

    print('Moving images to new closed-set dataset.')
    #move images that don't have the unknown classes, creating our new closed-set training dataset
    for dIdx, dName in enumerate(dNames):
        yr = dName.replace('Train', '').replace('Test', '')
        source_folder = f'{BASE_DATA_FOLDER}/VOCdevkit/VOC{yr}/JPEGImages/'
        destination_folder = f'{BASE_DATA_FOLDER}/VOCdevkit/VOC{yr}CS/JPEGImages/'

        #check destination folder exists, else create
        if not os.path.isdir(destination_folder):
            os.makedirs(destination_folder)

        for filename in tqdm.tqdm(filesIncluded[dName], total = len(filesIncluded[dName])):
            nm = filename.replace('JPEGImages/', '')
            shutil.copy(os.path.join(source_folder, nm), os.path.join(destination_folder, nm))

    #Fix annotations for closed-set training, validation and test dataset
    for yr in ['2007', '2012']:
        for split in ['trainval', 'train', 'val', 'test']:
            if yr == '2012' and split == 'test':
                continue #doesn't exist

            print(f'Changing annotation for VOC{yr} {split} split')
            source_file = f'{BASE_DATA_FOLDER}/VOCdevkit/VOC{yr}/ImageSets/Main/{split}.txt'
            destination_file = f'{BASE_DATA_FOLDER}/VOCdevkit/VOC{yr}CS/ImageSets/Main/{split}.txt'

            if not os.path.isdir(f'{BASE_DATA_FOLDER}/VOCdevkit/VOC{yr}CS/ImageSets/Main'):
                os.makedirs(f'{BASE_DATA_FOLDER}/VOCdevkit/VOC{yr}CS/ImageSets/Main')
            
            readFile = open(source_file, 'r')
            writeFile = open(destination_file, 'w')




            for x in readFile:
                xFormat = 'JPEGImages/'+x.replace('\n', '')+'.jpg'
                if yr == '2007':
                    if xFormat in filesIncluded['2007Train'] or xFormat in filesIncluded['2007Test']:
                        writeFile.write(x)
                else:
                    if xFormat in filesIncluded['2012Train']:
                        writeFile.write(x)
                    
            writeFile.close()

    print('Completed converting VOC to VOC-CS.')

elif args.dataset == 'coco':
    COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
        'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
        'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
        'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
        'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors',
        'teddy bear', 'hair drier', 'toothbrush')

    cocoData = dict(samples_per_gpu = 8, workers_per_gpu = 4,
                train = dict(type = 'CocoDataset',
                    classes=COCO_CLASSES,
                    ann_file=  BASE_DATA_FOLDER+'/coco/labels/annotations/instances_train2017.json',
                    img_prefix= BASE_DATA_FOLDER+'/coco/train2017/',
                    pipeline=img_pipeline),
                test = dict(type = 'CocoDataset',
                    classes=COCO_CLASSES,
                    ann_file=  BASE_DATA_FOLDER+'/coco/labels/annotations/instances_val2017.json',
                    img_prefix= BASE_DATA_FOLDER+'/coco/val2017/',
                    pipeline=img_pipeline))

    print('Building datasets')
    cocoTrainDataset = build_dataset(cocoData['train'])
    cocoTestDataset = build_dataset(cocoData['test'])


    imClasses = {'train': [], 'test': []}
    fileNames = {'train': [], 'test': []}
    numIgnores = {'train': [], 'test': []}

    cocoDatasets = {'train': cocoTrainDataset, 'test': cocoTestDataset}

    for split in ['train', 'test']:
        for imIdx in range(len(cocoDatasets[split])):
            imInfo = cocoDatasets[split].get_ann_info(imIdx)
            #ignore bboxes are always crowds, which will be class person, which we will always include because it is a known class
            clsesPresent = list(imInfo['labels']) 
            imClasses[split] += [clsesPresent]
            fileNames[split] += [cocoDatasets[split].data_infos[imIdx]['filename']]
            numIgnores[split] += [len(imInfo['bboxes_ignore'])]

    # For COCO, first 40 classes are known (1-40), rest are unknown
    cutoffCls = 40

    totalInstances = {'train':[1 for i in range(81)], 'test':[1 for i in range(81)]}
    includedInstances = {'train':[1 for i in range(81)], 'test':[1 for i in range(81)]}
    namesIncluded = {'train': [], 'test': []}
    for split in ['train', 'test']:
        for imIdx, imCls in enumerate(imClasses[split]):
            #statistics of original instance distribution in training data
            for cl in imCls:
                totalInstances[split][cl] += 1

            mask = np.asarray(imCls) > cutoffCls
            if np.sum(mask) != 0:
                continue
                
            namesIncluded[split] += [fileNames[split][imIdx]]
            #statistics of original instance distribution in training data
            for cl in imCls:
                includedInstances[split][cl] += 1

    #check the distribution of the new training dataset
    #let's check the data balance in our new training and test dataset

    plt.figure()
    plt.plot(includedInstances['train'], label = f'New training dataset')
    plt.plot(totalInstances['train'], label = f'Original training dataset')
    plt.ylabel('Number of instances')
    plt.xlabel('Class ID')
    plt.legend()
    plt.title(f'Number of class instances between original and new training dataset')
    plt.show()

    #as a percent of original training dataset?
    percentInstances = np.array(includedInstances['train'])/np.array(totalInstances['train'])
    plt.figure()
    plt.plot(percentInstances)
    plt.ylabel('% class instances retained')
    plt.xlabel('Class ID')
    plt.title(f'Percent of class instances retained in new training dataset')
    plt.show()


    #split the training data into training data and validation data
    totalTrainIms = len(namesIncluded['train'])
    newTrainIms = totalTrainIms*0.8
    namesIncluded['trainNew'] = []
    namesIncluded['val'] = []
    includedInstancesTrain = [0 for i in range(80)]
    
    count = 0
    for idx, imCls in enumerate(imClasses['train']):
        mask = np.asarray(imCls) > cutoffCls
        if np.sum(mask) != 0:
            continue
        if count <= newTrainIms:
            for cl in imCls:
                includedInstancesTrain[cl] += 1
            namesIncluded['trainNew'] += [fileNames['train'][idx]]
        else:
            namesIncluded['val'] += [fileNames['train'][idx]]
        count += 1
        
    plt.figure()
    plt.plot(np.array(includedInstancesTrain)/np.array(includedInstances['train']))
    plt.xlabel('Class ID')
    plt.ylabel('Percent in new training split - 80% desired')
    plt.title('Class Instances split between training and validation dataset')
    plt.show()

    print('Moving images to new closed-set folder')
    source_folders = [BASE_DATA_FOLDER+'/coco/images/train2017/', BASE_DATA_FOLDER+'/coco/images/train2017/', BASE_DATA_FOLDER+'/coco/images/val2017/']
    destination_folders = [BASE_DATA_FOLDER+'/coco/images/'+split+'CS2017/' for split in ['train', 'val', 'test']]

    #check destination folder exists, else create
    for folder in destination_folders:
        if not os.path.isdir(folder):
            os.makedirs(folder)

    for spIdx, split in enumerate(['trainNew', 'val', 'test']):
        for filename in tqdm.tqdm(namesIncluded[split]):
            shutil.copy(os.path.join(source_folders[spIdx], filename), os.path.join(destination_folders[spIdx], filename))

    #fix annotations
    source_fileTrain = BASE_DATA_FOLDER+'/coco/labels/annotations/instances_train2017.json'
    source_fileTest = BASE_DATA_FOLDER+'/coco/labels/annotations/instances_val2017.json'
    
    destination_fileTrain = BASE_DATA_FOLDER+'/coco/labels/annotations/instances_trainCS2017.json'
    destination_fileVal = BASE_DATA_FOLDER+'/coco/labels/annotations/instances_valCS2017.json'
    destination_fileTest = BASE_DATA_FOLDER+'/coco/labels/annotations/instances_testCS2017.json'
    destination_files = {'trainNew': destination_fileTrain, 'val': destination_fileVal, 'test': destination_fileTest}


    with open(source_fileTrain) as f:
        readFileTrain = json.load(f)
    with open(source_fileTest) as f:
        readFileTest = json.load(f)

    readFiles = [readFileTrain, readFileTrain, readFileTest]
    

    for spIdx, split in enumerate(['trainNew', 'val', 'test']):
        print(f'Changing annotations for {split} split')
        readFile = readFiles[spIdx]

        writeFile = {}

        writeFile['info'] = readFile['info']
        writeFile['licenses'] = readFile['licenses']

        writeFile['categories'] = readFile['categories'][:50]

        writeFile['images'] = []
        writeFile['annotations'] = []

        for imKey in tqdm.tqdm(readFile['images']):
            name = imKey['file_name']
            if name in namesIncluded[split]:
                writeFile['images'] += [imKey]

        for annKey in tqdm.tqdm(readFile['annotations']):
            category = annKey['category_id']
            if category > 40: # this corresponds to 40th class in coco
                continue
            name = str(annKey['image_id']).zfill(12) + '.jpg'
            if name in namesIncluded[split]:
                writeFile['annotations'] += [annKey]


        with open(destination_files[split], 'w') as outFile:
            json.dump(writeFile, outFile)

        print('Completed converting COCO to COCO-CS.')

elif args.dataset == 'bdd':
    BDD_CLASSES = ('pedestrian','rider','car','truck','bus','train','motorcycle','bicycle','traffic light','traffic sign')
    bddData = dict(samples_per_gpu = 8, workers_per_gpu = 4,
                train = dict(type = 'CocoDataset',
                    classes=BDD_CLASSES,
                    ann_file= '/home/root/bdd100k/train_coco_format.json',
                    img_prefix= '/home/root/bdd100k/images/100k/train/',
                    pipeline=img_pipeline),
                test = dict(type = 'CocoDataset',
                    classes=BDD_CLASSES,
                    ann_file= '/home/root/bdd100k/val_coco_format.json',
                    img_prefix= '/home/root/bdd100k/images/100k/test/',
                    pipeline=img_pipeline),
                val = dict(type = 'CocoDataset',
                    classes=BDD_CLASSES,
                    ann_file= '/home/root/bdd100k/val_coco_format.json',
                    img_prefix= '/home/root/bdd100k/images/100k/val/',
                    pipeline=img_pipeline))

    print('Building datasets')
    bddTrainDataset = build_dataset(bddData['train'])
    bddTestDataset = build_dataset(bddData['test'])
    bddValDataset = build_dataset(bddData['val'])

    imClasses = {'train': [], 'test': [], 'val': []}
    fileNames = {'train': [], 'test': [], 'val': []}
    numIgnores = {'train': [], 'test': [], 'val': []}

    bddDatasets = {'train': bddTrainDataset, 'test': bddTestDataset, 'val': bddValDataset}

    for split in ['train', 'test', 'val']:
        for imIdx in range(len(bddDatasets[split])):
            imInfo = bddDatasets[split].get_ann_info(imIdx)
            #ignore bboxes are always crowds, which will be class person, which we will always include because it is a known class
            clsesPresent = list(imInfo['labels']) 
            imClasses[split] += [clsesPresent]
            fileNames[split] += [bddDatasets[split].data_infos[imIdx]['filename']]
            numIgnores[split] += [len(imInfo['bboxes_ignore'])]

    #For BDD100K, first 6 classes are known (1-6), rest are unknown
    cutoffCls = 6

    totalInstances = {'train':[1 for i in range(11)], 'test':[1 for i in range(11)], 'val':[1 for i in range(11)]}
    includedInstances = {'train':[1 for i in range(11)], 'test':[1 for i in range(11)], 'val':[1 for i in range(11)]}
    namesIncluded = {'train': [], 'test': [], 'val': []}
    for split in ['train', 'test', 'val']:
        for imIdx, imCls in enumerate(imClasses[split]):
            #statistics of original instance distribution in training data
            for cl in imCls:
                totalInstances[split][cl] += 1

            mask = np.asarray(imCls) > cutoffCls
            if np.sum(mask) != 0:
                continue
                
            namesIncluded[split] += [fileNames[split][imIdx]]
            #statistics of original instance distribution in training data
            for cl in imCls:
                includedInstances[split][cl] += 1

    #check the distribution of the new training dataset
    #let's check the data balance in our new training and test dataset

    plt.figure()
    plt.plot(includedInstances['train'], label = f'New training dataset')
    plt.plot(totalInstances['train'], label = f'Original training dataset')
    plt.ylabel('Number of instances')
    plt.xlabel('Class ID')
    plt.legend()
    plt.title(f'Number of class instances between original and new training dataset')
    plt.show()

    #as a percent of original training dataset?
    percentInstances = np.array(includedInstances['train'])/np.array(totalInstances['train'])
    plt.figure()
    plt.plot(percentInstances)
    plt.ylabel('% class instances retained')
    plt.xlabel('Class ID')
    plt.title(f'Percent of class instances retained in new training dataset')
    plt.show()


    #split the training data into training data and validation data
    totalTrainIms = len(namesIncluded['train'])
    newTrainIms = totalTrainIms*0.8
    namesIncluded['trainNew'] = []
    namesIncluded['val'] = []
    includedInstancesTrain = [0 for i in range(10)]
    
    count = 0
    for idx, imCls in enumerate(imClasses['train']):
        mask = np.asarray(imCls) > cutoffCls
        if np.sum(mask) != 0:
            continue
        if count <= newTrainIms:
            for cl in imCls:
                includedInstancesTrain[cl] += 1
            namesIncluded['trainNew'] += [fileNames['train'][idx]]
        else:
            namesIncluded['val'] += [fileNames['train'][idx]]
        count += 1
        
    plt.figure()
    plt.plot(np.array(includedInstancesTrain)/np.array(includedInstances['train']))
    plt.xlabel('Class ID')
    plt.ylabel('Percent in new training split - 80% desired')
    plt.title('Class Instances split between training and validation dataset')
    plt.show()

    print('Moving images to new closed-set folder')
    source_folders = ['/home/root/bdd100k/images/100k/train/', '/home/root/bdd100k/images/100k/train/', '/home/root/bdd100k/images/100k/val/']
    destination_folders = ['/home/root/bdd100k/images/100k/'+split+'CS/' for split in ['train', 'val', 'test']]

    #check destination folder exists, else create
    for folder in destination_folders:
        if not os.path.isdir(folder):
            os.makedirs(folder)

    for spIdx, split in enumerate(['trainNew', 'val', 'test']):
        for filename in tqdm.tqdm(namesIncluded[split]):
            shutil.copy(os.path.join(source_folders[spIdx], filename), os.path.join(destination_folders[spIdx], filename))

    #fix annotations
    source_fileTrain = '/home/root/bdd100k/train_coco_format.json'
    source_fileTest = '/home/root/bdd100k/val_coco_format.json'
    
    destination_fileTrain = '/home/root/bdd100k/trainCS_coco_format.json'
    destination_fileVal = '/home/root/bdd100k/valCS_coco_format.json'
    destination_fileTest = '/home/root/bdd100k/testCS_coco_format.json'
    destination_files = {'trainNew': destination_fileTrain, 'val': destination_fileVal, 'test': destination_fileTest}


    with open(source_fileTrain) as f:
        readFileTrain = json.load(f)
    with open(source_fileTest) as f:
        readFileTest = json.load(f)

    readFiles = [readFileTrain, readFileTrain, readFileTest]
    

    for spIdx, split in enumerate(['trainNew', 'val', 'test']):
        print(f'Changing annotations for {split} split')
        readFile = readFiles[spIdx]

        writeFile = {}

        writeFile['info'] = readFile['info']
        writeFile['licenses'] = readFile['licenses']

        writeFile['categories'] = readFile['categories'][:50]

        writeFile['images'] = []
        writeFile['annotations'] = []

        for imKey in tqdm.tqdm(readFile['images']):
            name = imKey['file_name']
            if name in namesIncluded[split]:
                writeFile['images'] += [imKey]

        for annKey in tqdm.tqdm(readFile['annotations']):
            category = annKey['category_id']
            if category > 55: # this corresponds to 50th class in coco
                continue
            name = str(annKey['image_id']).zfill(12) + '.jpg'
            if name in namesIncluded[split]:
                writeFile['annotations'] += [annKey]


        with open(destination_files[split], 'w') as outFile:
            json.dump(writeFile, outFile)
        
        print('Completed converting BDD to BDD-CS.')

elif args.dataset == 'kitti':
    KITTI_CLASSES = ('car', 'van', 'truck','pedestrian', 'person_sitting', 'cyclist', 'tram', 'misc', 'dontCare')

    kittiData = dict(samples_per_gpu = 8, workers_per_gpu = 4,
                train = dict(type = 'CocoDataset',
                    classes=KITTI_CLASSES,
                    ann_file= '/home/root/kitti/object/training/label2-COCO-Format/train_coco_format.json',
                    img_prefix= '/home/root/kitti/object/training/image_2/',
                    pipeline=img_pipeline),
                test = dict(type = 'CocoDataset',
                    classes=KITTI_CLASSES,
                    ann_file=  '/home/root/kitti/object/training/label2-COCO-Format/val_coco_format.json',
                    img_prefix= '/home/root/kitti/object/testing/image_2/',
                    pipeline=img_pipeline))

    print('Building datasets')
    kittiTrainDataset = build_dataset(kittiData['train'])
    kittiTestDataset = build_dataset(kittiData['test'])


    imClasses = {'train': [], 'test': []}
    fileNames = {'train': [], 'test': []}
    numIgnores = {'train': [], 'test': []}

    kittiDatasets = {'train': kittiTrainDataset, 'test': kittiTestDataset}

    for split in ['train', 'test']:
        for imIdx in range(len(kittiDatasets[split])):
            imInfo = kittiDatasets[split].get_ann_info(imIdx)
            #ignore bboxes are always crowds, which will be class person, which we will always include because it is a known class
            clsesPresent = list(imInfo['labels']) 
            imClasses[split] += [clsesPresent]
            fileNames[split] += [kittiDatasets[split].data_infos[imIdx]['filename']]
            numIgnores[split] += [len(imInfo['bboxes_ignore'])]

    # For KITTI, first 6 classes are known (1-6), rest are unknown
    cutoffCls = 6

    totalInstances = {'train':[1 for i in range(10)], 'test':[1 for i in range(10)]}
    includedInstances = {'train':[1 for i in range(10)], 'test':[1 for i in range(10)]}
    namesIncluded = {'train': [], 'test': []}
    for split in ['train', 'test']:
        for imIdx, imCls in enumerate(imClasses[split]):
            #statistics of original instance distribution in training data
            for cl in imCls:
                totalInstances[split][cl] += 1

            mask = np.asarray(imCls) > cutoffCls
            if np.sum(mask) != 0:
                continue
                
            namesIncluded[split] += [fileNames[split][imIdx]]
            #statistics of original instance distribution in training data
            for cl in imCls:
                includedInstances[split][cl] += 1

    #check the distribution of the new training dataset
    #let's check the data balance in our new training and test dataset

    plt.figure()
    plt.plot(includedInstances['train'], label = f'New training dataset')
    plt.plot(totalInstances['train'], label = f'Original training dataset')
    plt.ylabel('Number of instances')
    plt.xlabel('Class ID')
    plt.legend()
    plt.title(f'Number of class instances between original and new training dataset')
    plt.show()

    #as a percent of original training dataset?
    percentInstances = np.array(includedInstances['train'])/np.array(totalInstances['train'])
    plt.figure()
    plt.plot(percentInstances)
    plt.ylabel('% class instances retained')
    plt.xlabel('Class ID')
    plt.title(f'Percent of class instances retained in new training dataset')
    plt.show()


    #split the training data into training data and validation data
    totalTrainIms = len(namesIncluded['train'])
    newTrainIms = totalTrainIms*0.8
    namesIncluded['trainNew'] = []
    namesIncluded['val'] = []
    includedInstancesTrain = [0 for i in range(9)]
    
    count = 0
    for idx, imCls in enumerate(imClasses['train']):
        mask = np.asarray(imCls) > cutoffCls
        if np.sum(mask) != 0:
            continue
        if count <= newTrainIms:
            for cl in imCls:
                includedInstancesTrain[cl] += 1
            namesIncluded['trainNew'] += [fileNames['train'][idx]]
        else:
            namesIncluded['val'] += [fileNames['train'][idx]]
        count += 1
        
    plt.figure()
    plt.plot(np.array(includedInstancesTrain)/np.array(includedInstances['train']))
    plt.xlabel('Class ID')
    plt.ylabel('Percent in new training split - 80% desired')
    plt.title('Class Instances split between training and validation dataset')
    plt.show()

    print('Moving images to new closed-set folder')
    source_folders = ['/home/root/kitti/object/training/image_2/', '/home/root/kitti/object/training/image_2/', '/home/root/kitti/object/testing/image_2/']
    destination_folders = ['/home/root/kitti/object/'+split+'CS2017/' for split in ['train', 'val', 'test']]

    #check destination folder exists, else create
    for folder in destination_folders:
        if not os.path.isdir(folder):
            os.makedirs(folder)

    for spIdx, split in enumerate(['trainNew', 'val', 'test']):
        for filename in tqdm.tqdm(namesIncluded[split]):
            shutil.copy(os.path.join(source_folders[spIdx], filename), os.path.join(destination_folders[spIdx], filename))

    #fix annotations
    source_fileTrain = '/home/root/kitti/object/training/label2-COCO-Format/train_coco_format.json'
    source_fileTest = '/home/root/kitti/object/training/label2-COCO-Format/val_coco_format.json'
    
    destination_fileTrain = '/home/root/kitti/object/training/label2-COCO-Format/trainCS_coco_format.json'
    destination_fileVal = '/home/root/kitti/object/training/label2-COCO-Format/valCS_coco_format.json'
    destination_fileTest = '/home/root/kitti/object/training/label2-COCO-Format/testCS_coco_format.json'
    destination_files = {'trainNew': destination_fileTrain, 'val': destination_fileVal, 'test': destination_fileTest}


    with open(source_fileTrain) as f:
        readFileTrain = json.load(f)
    with open(source_fileTest) as f:
        readFileTest = json.load(f)

    readFiles = [readFileTrain, readFileTrain, readFileTest]
    

    for spIdx, split in enumerate(['trainNew', 'val', 'test']):
        print(f'Changing annotations for {split} split')
        readFile = readFiles[spIdx]

        writeFile = {}

        writeFile['info'] = readFile['info']
        writeFile['licenses'] = readFile['licenses']

        writeFile['categories'] = readFile['categories'][:50]

        writeFile['images'] = []
        writeFile['annotations'] = []

        for imKey in tqdm.tqdm(readFile['images']):
            name = imKey['file_name']
            if name in namesIncluded[split]:
                writeFile['images'] += [imKey]

        for annKey in tqdm.tqdm(readFile['annotations']):
            category = annKey['category_id']
            if category > 10: # this corresponds to 50th class in coco
                continue
            name = str(annKey['image_id']).zfill(12) + '.jpg'
            if name in namesIncluded[split]:
                writeFile['annotations'] += [annKey]


        with open(destination_files[split], 'w') as outFile:
            json.dump(writeFile, outFile)

        print('Completed converting KITTI to KITTI-CS.')

elif args.dataset == 'idd':
    IDD_CLASSES = ('car', 'person','rider','truck','motorcycle','bus', 'bicycle','train','caravan','trailer','animal','traffic sign','traffic light')
    iddData = dict(samples_per_gpu = 1, workers_per_gpu = 4,
                train = dict(type = 'XMLDataset',
                    classes=IDD_CLASSES,
                    ann_file= '/home/root/IDD_Detection/train.txt',
                    img_prefix= '/home/root/IDD_Detection/',
                    pipeline=img_pipeline),
                val = dict(type = 'XMLDataset',
                    classes=IDD_CLASSES,
                    ann_file=  '/home/root/IDD_Detection/val.txt',
                    img_prefix= '/home/root/IDD_Detection/',
                    pipeline=img_pipeline))
    
    print('Building datasets')
    iddDatasetTrain = build_dataset(iddData['train'])
    #iddDatasetTest = build_dataset(iddData['test'])
    iddDatasetVal = build_dataset(iddData['val'])

    #for each dataset, collect classes in each image and their filenames
    imClasses = {'Train': [], 'Val': []}
    fileNames = {'Train': [], 'Val': []}

    dNames = ['Train', 'Val']
    for dIdx, dataset in enumerate([iddDatasetTrain, iddDatasetVal]):
        for imIdx in range(len(dataset)):
            imInfo = dataset.get_ann_info(imIdx)
            clsesPresent = list(imInfo['labels']) + list(imInfo['labels_ignore'])
            imClasses[dNames[dIdx]] += [clsesPresent]
            fileNames[dNames[dIdx]] += [dataset.data_infos[imIdx]['filename']]


    # For IDD, the first 9 classes (0-8) are 'known' classes, and the rest are 'unknown'
    cutoffCls = 8

    totalTrainImgs = len(iddDatasetTrain)
    includedTrainImgs = 0
    totalTrainInstances = [0 for i in range(13)]
    includedTrainInstances = [0 for i in range(13)]
    filesIncluded = {'Train': [], 'Val':[]}

    for dIdx, dName in enumerate(dNames):
        for imIdx, imCls in enumerate(imClasses[dName]):
            #statistics of original instance distribution in training data
            if 'Train' in dName:
                for cl in imCls:
                    totalTrainInstances[cl] += 1

            mask = np.asarray(imCls) > cutoffCls

            #if the image has any 'unknown' classes, it is not included in the new training dataset
            if np.sum(mask) != 0:
                continue

            #otherwise it is included in the new training dataset
            filesIncluded[dName] += [fileNames[dName][imIdx]]

            #statistics of new instance distribution in training data
            if 'Train' in dName:
                for cl in imCls:
                    includedTrainInstances[cl] += 1
                includedTrainImgs += 1

    #let's check the data balance in our new training dataset
    plt.figure()
    plt.plot(includedTrainInstances, label = 'New Training dataset')
    plt.plot(totalTrainInstances, label = 'Original Training dataset')
    plt.ylabel('Number of instances')
    plt.xlabel('Class ID')
    plt.legend()
    plt.title('Number of class instances between original and new training dataset')
    plt.show()

    #data balance as a percent of original training dataset?
    percentInstances = np.array(includedTrainInstances)/np.array(totalTrainInstances)
    plt.figure()
    plt.plot(percentInstances)
    plt.ylabel('% class instances retained')
    plt.xlabel('Class ID')
    plt.title('Percent of class instances retained in new training dataset')
    plt.show()

    print('Moving images to new closed-set dataset.')
    #move images that don't have the unknown classes, creating our new closed-set training dataset
    for dIdx, dName in enumerate(dNames):
        #yr = dName.replace('Train', '').replace('Test', '')
        source_folder = f'/home/root/IDD_Detection/JPEGImages/'
        destination_folder = f'/home/root/IDD_DetectionCS/JPEGImages/'

        #check destination folder exists, else create
        if not os.path.isdir(destination_folder):
            os.makedirs(destination_folder)

        for filename in tqdm.tqdm(filesIncluded[dName], total = len(filesIncluded[dName])):
            nm = filename.replace('JPEGImages/', '')
            shutil.copy(os.path.join(source_folder, nm), os.path.join(destination_folder, nm))

    #Fix annotations for closed-set training, validation and test dataset
    #for yr in ['2007', '2012']:
        for split in ['train', 'val', 'test']:
            #if yr == '2012' and split == 'test':
            #    continue #doesn't exist

            print(f'Changing annotation for IDD {split} split')
            source_file = f'/home/root/IDD_Detection/{split}.txt'
            destination_file = f'/home/root/IDD_DetectionCS/{split}.txt'

            if not os.path.isdir(f'/home/root/IDD_DetectionCS'):
                os.makedirs(f'/home/root/IDD_DetectionCS')
            
            readFile = open(source_file, 'r')
            writeFile = open(destination_file, 'w')

            for x in readFile:
                xFormat = 'JPEGImages/'+x.replace('\n', '')+'.jpg'
                #if yr == '2007':
                if xFormat in filesIncluded['Train'] or xFormat in filesIncluded['Val']:
                    writeFile.write(x)
                #else:
                    #if xFormat in filesIncluded['2012Train']:
                    #    writeFile.write(x)
                    
            writeFile.close()

    print('Completed converting IDD to IDD-CS.')

else:
    print('This dataset is not implemented.')
    exit()

