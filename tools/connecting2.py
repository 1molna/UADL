import json

import argparse

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
						 wrap_fp16_model)

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
							replace_ImageToTensor)
from mmdet.models import build_detector

import numpy as np
import cv2
import tqdm
import os
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from base_dirs import *

def parse_args():
	parser = argparse.ArgumentParser(description='Test with Distance')
	parser.add_argument('dType', default = 'FRCNN', help='FRCNN or retinanet')
	parser.add_argument('--dataset', default = 'voc', help='voc or coco or bdd or kitti or idd')
	parser.add_argument('--train', default = 1, type = int, help='Connect training data')
	parser.add_argument('--test', default = 1, type = int, help='Connect test data')
	parser.add_argument('--val', default = 1, type = int, help='Connect validation data')
	parser.add_argument('--saveNm', default = None, help='the save name of the raw results')
	args = parser.parse_args()
	return args

args = parse_args()


#load the config file for the model
if args.dataset == 'voc':
	args.config = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_vocOS_f.py'
	num_classes = 15
elif args.dataset == 'bdd':
	args.config = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_bddOS_f.py'
	num_classes = 6
elif args.dataset == 'kitti':
	args.config = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_kittiOS_f.py'
	num_classes = 6
elif args.dataset == 'idd':
	args.config = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_iddOS_f.py'
	num_classes = 9
else:
	args.config = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_cocoOS_wLogits2.py'
	num_classes = 40

cfg = Config.fromfile(args.config)

# import modules from string list.
if cfg.get('custom_imports', None):
	from mmcv.utils import import_modules_from_strings
	import_modules_from_strings(**cfg['custom_imports'])
# set cudnn_benchmark
if cfg.get('cudnn_benchmark', False):
	torch.backends.cudnn.benchmark = True
cfg.model.pretrained = None
if cfg.model.get('neck'):
	if isinstance(cfg.model.neck, list):
		for neck_cfg in cfg.model.neck:
			if neck_cfg.get('rfp_backbone'):
				if neck_cfg.rfp_backbone.get('pretrained'):
					neck_cfg.rfp_backbone.pretrained = None
	elif cfg.model.neck.get('rfp_backbone'):
		if cfg.model.neck.rfp_backbone.get('pretrained'):
			cfg.model.neck.rfp_backbone.pretrained = None

# in case the test dataset is concatenated
if isinstance(cfg.data.testOS, dict):
	cfg.data.testOS.test_mode = True
elif isinstance(cfg.data.testOS, list):
	for ds_cfg in cfg.data.testOS:
		ds_cfg.test_mode = True

distributed = False

samples_per_gpu = cfg.data.testOS.pop('samples_per_gpu', 1)
if samples_per_gpu > 1:
	# Replace 'ImageToTensor' to 'DefaultFormatBundle'
	cfg.data.testOS.pipeline = replace_ImageToTensor(cfg.data.testOS.pipeline)



#iou threshold for object to be Connected with detection
iouThresh = 0.5
#score threshold for detection to be considered valid
scoreThresh = 0.2

#function used to calculate IoU between boxes, taken from: https://medium.com/@venuktan/vectorized-intersection-over-union-iou-in-numpy-and-tensor-flow-4fa16231b63d
def iouCalc(boxes1, boxes2):
	def run(bboxes1, bboxes2):
		x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
		x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
		xA = np.maximum(x11, np.transpose(x21))
		yA = np.maximum(y11, np.transpose(y21))
		xB = np.minimum(x12, np.transpose(x22))
		yB = np.minimum(y12, np.transpose(y22))
		interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
		boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
		boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
		iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
		return iou
	return run(boxes1, boxes2)

#used to Connect detections either as background, known class correctly predicted, known class incorrectly predicted, unknown class
def Connect_detections(dataHolder, dets, gt, clsCutoff = 6):
		gtBoxes = gt['bboxes']
		gtLabels = gt['labels']
		detPredict = dets['predictions']
		detBoxes = dets['boxes']
		detScores = dets['scores']
		detF = dets['features']
		

		knownBoxes = gtBoxes[gtLabels < clsCutoff]
		knownLabels = gtLabels[gtLabels < clsCutoff]
		unknownBoxes = gtBoxes[gtLabels > clsCutoff]
		unknownLabels = gtLabels[gtLabels > clsCutoff]

		#sort from most confident to least
		sorted_scores = np.sort(detScores)[::-1]
		sorted_idxes = np.argsort(detScores)[::-1]

		detConnected = [0]*len(detScores)
		gtKnownConnected = [0]*len(knownBoxes)

		#first, we check if the detection has fallen on a known class
		#if an IoU > iouThresh with a known class --> it is detecting that known class
		if len(knownBoxes) > 0:
			knownIous = iouCalc(detBoxes, knownBoxes)

			for idx, score in enumerate(sorted_scores):
				#if all gt have been Connected, move on
				if np.sum(gtKnownConnected) == len(gtKnownConnected):
					break

				detIdx = sorted_idxes[idx]
				ious = knownIous[detIdx]
				#sort from greatest to lowest overlap
				sorted_iouIdxs = np.argsort(ious)[::-1]
				
				for iouIdx in sorted_iouIdxs:
					#check this gt object hasn't already been detected
					if gtKnownConnected[iouIdx] == 1:
						continue

					if ious[iouIdx] >= iouThresh:
						#associating this detection and gt object
						gtKnownConnected[iouIdx] = 1
						detConnected[detIdx] = 1

						gtLbl = knownLabels[iouIdx]
						dataHolder['ious'] += [ious[iouIdx]]
						#known class was classified correctly
						if detPredict[detIdx] == gtLbl:
							dataHolder['scores'] += [score]
							dataHolder['features'] += [list(detF[detIdx])]
							dataHolder['type'] += [0]
						#known class was misclassified
						else:
							dataHolder['scores'] += [score]
							dataHolder['features'] += [list(detF[detIdx])]
							dataHolder['type'] += [1]
						break
					else:
						#doesn't have an iou greater than 0.5 with anything
						break
		
		#all detections have been Connected
		if np.sum(detConnected) == len(detConnected):
			return dataHolder

		### Next, check if the detection overlaps an ignored gt known object - these detections are ignored
		#also check ignored gt known objects
		if len(gt['bboxes_ignore']) > 0:
			igBoxes = gt['bboxes_ignore']
			igIous = iouCalc(detBoxes, igBoxes)
			for idx, score in enumerate(sorted_scores):
				detIdx = sorted_idxes[idx]
				if detConnected[detIdx] == 1:
					continue

				ious = igIous[detIdx]

				#sort from greatest to lowest overlap
				sorted_iouIdxs = np.argsort(ious)[::-1]

				for iouIdx in sorted_iouIdxs:
					if ious[iouIdx] >= iouThresh:
						#associating this detection and gt object
						detConnected[detIdx] = 1
					break


		#all detections have been Connected
		if np.sum(detConnected) == len(detConnected):
			return dataHolder

		#if an IoU > 0.5 with an unknown class (but not any known classes) --> it is detecting the unknown class
		newDetConnected = detConnected
		if len(unknownBoxes) > 0:
			unknownIous = iouCalc(detBoxes, unknownBoxes)

			for idx, score in enumerate(sorted_scores):
				detIdx = sorted_idxes[idx]

				#if the detection has already been Connected, skip it
				if detConnected[detIdx] == 1:
					continue

				ious = unknownIous[detIdx]

				#sort from greatest to lowest overlap
				sorted_iouIdxs = np.argsort(ious)[::-1]
				for iouIdx in sorted_iouIdxs:
					if ious[iouIdx] >= iouThresh:
						newDetConnected[detIdx] = 1

						gtLbl = unknownLabels[iouIdx]
						dataHolder['scores'] += [score]
						dataHolder['features'] += [list(detF[detIdx])]
						dataHolder['type'] += [2]
						dataHolder['ious'] += [ious[iouIdx]]
						break
					else:
						#no overlap greater than 0.5
						break


		detConnected = newDetConnected

		if np.sum(detConnected) == len(detConnected):
			return dataHolder

		#otherwise remaining detections are all background detections
		for detIdx, assoc in enumerate(detConnected):
			if not assoc:
				score = detScores[detIdx]
				dataHolder['scores'] += [score]
				dataHolder['type'] += [3]
				dataHolder['features'] += [list(detF[detIdx])]
				dataHolder['ious'] += [0]
				detConnected[detIdx] = 1

		if np.sum(detConnected) != len(detConnected):
			print("THERE IS A BIG CONNECTION PROBLEM")
			exit()
		
		return dataHolder


results_dir = f'./{args.dType}/raw/{args.dataset}'
save_dir = f'./{args.dType}/Connected/{args.dataset}'

if args.train:
	print('Associating training data')
	allfeatures = []
	allLabels = []
	allScores = []
	allIoUs = []

	if args.dataset == 'voc':
		trainDataset07 = build_dataset(cfg.data.trainCS07)
		trainDataset12 = build_dataset(cfg.data.trainCS12)
		trainDatasets = [trainDataset07, trainDataset12]
		
		with open(f'{results_dir}/train07/{args.saveNm}.json', 'r') as f:
			train07Dets = json.load(f)

		with open(f'{results_dir}/train12/{args.saveNm}.json', 'r') as f:
			train12Dets = json.load(f)

		allTrainDets = [train07Dets, train12Dets]
	else:

		trainDataset = build_dataset(cfg.data.trainT)
		trainDatasets = [trainDataset]

		with open(f'{results_dir}/train/{args.saveNm}.json', 'r') as f:
			trainDets = json.load(f)

		allTrainDets = [trainDets]

	for tIdx, trainDataset in enumerate(trainDatasets):
		trainDets = allTrainDets[tIdx]
		lenDataset = len(trainDataset)
		for imIdx, im in enumerate(tqdm.tqdm(trainDataset, total = lenDataset)):
			imName = trainDataset.data_infos[imIdx]['filename']	
			
			detData = np.asarray(trainDets[imName])
			gtData = trainDataset.get_ann_info(imIdx)

			#continue if no detections made
			if len(detData) == 0:
				continue

			detBoxes = detData[:, -5:-1]
			detScores = detData[:, -1]
			detF = detData[:, :-5]
			detPredict = np.argmax(detF, axis = 1)

			if args.dType == 'retinanet':
				if 'Ensembles' not in args.saveNm:
					newdetF = np.log(detF/(1-detF))
				else:
					newdetF = detF
					detF = 1/(1+np.exp(-newdetF))
				mask = np.max(newdetF, axis = 1) > 100
				if np.sum(mask) > 0:
					if np.sum(mask) > 1:
						print("ISSUE")
						exit()

					idxes = np.where(detF == 1)
					idx1 = idxes[0][0]
					idx2 = idxes[1][0]
					newdetF[idx1, idx2] = 25
				detF = newdetF


			gtBoxes = gtData['bboxes']
			gtLabels = gtData['labels']
			
			ious = iouCalc(detBoxes, gtBoxes)
			for detIdx, guess in enumerate(detPredict):
				iou = ious[detIdx]
				mask = iou > iouThresh

				trueClasses = gtLabels[mask]
				gtMatches = np.where(guess == trueClasses)[0]

				if len(gtMatches) > 0:
					allfeatures += [detF[detIdx].tolist()]
					allLabels += [int(guess)]
					allScores += [detScores[detIdx]]

					maxIoU = np.max(iou[mask][gtMatches])
					allIoUs += [maxIoU]
			
	allfeatures = list(allfeatures)
	allLabels = list(allLabels)
	allScores = list(allScores)
	allIoUs = list(allIoUs)

	trainDict = {'features': allfeatures, 'labels': allLabels, 'scores': allScores, 'ious': allIoUs}

	sub_save_dir = save_dir+f'/train/'
	if not os.path.exists(sub_save_dir):
		os.makedirs(sub_save_dir)
	with open(f'{sub_save_dir}{args.saveNm}.json', 'w') as outFile:
		json.dump(trainDict, outFile)


for typIdx, nm in enumerate(['val', 'test']):
	if nm == 'val':
		if not bool(args.val):
			continue
		testDataset = build_dataset(cfg.data.val)
	else:
		if not bool(args.test):
			continue
		testDataset = build_dataset(cfg.data.testOS)

	
	print(f'Associating {nm} data')
	allData = {'scores': [], 'type': [], 'features': [], 'ious': []}
	lenDataset = len(testDataset)

	with open(f'{results_dir}/{nm}/{args.saveNm}.json', 'r') as f:
		testDets = json.load(f)

	for imIdx, im in enumerate(tqdm.tqdm(testDataset, total = lenDataset)):
		imName = testDataset.data_infos[imIdx]['filename']
		
		detData = np.asarray(testDets[imName])
		gtData = testDataset.get_ann_info(imIdx)
		
		if len(detData) == 0: #no detections for this image
				continue
		
		detF = detData[:, :-5]
		detBoxes = detData[:, -5:-1] 	
		detScores = detData[:, -1]
		detPredict = np.argmax(detF, axis = 1)

		if args.dType == 'retinanet':
			if 'Ensembles' not in args.saveNm and 'Ens' not in args.saveNm:
				newdetF = np.log(detF/(1-detF))
			else:
				newdetF = detF
				detF = 1/(1+np.exp(-newdetF))
			
			mask = np.max(newdetF, axis = 1) > 25
			if np.sum(mask) > 0:
				if np.sum(mask) > 1:
					print("ISSUE")
					exit()

				idxes = np.where(detF == 1)
				idx1 = idxes[0][0]
				idx2 = idxes[1][0]
				newdetF[idx1, idx2] = 25
			detF = newdetF

		#only consider detections that meet the score threshold
		mask = detScores >= scoreThresh
		detScores = detScores[mask]
		detBoxes = detBoxes[mask]
		detF = detF[mask]
		detPredict = detPredict[mask]

		allDetsIm = {'predictions': detPredict, 'scores': detScores, 'boxes': detBoxes, 'features': detF}
		
		#Connect detections to objects
		allData = Connect_detections(allData, allDetsIm, gtData, clsCutoff = num_classes)

	sub_save_dir = save_dir+f'/{nm}/'
	if not os.path.exists(sub_save_dir):
		os.makedirs(sub_save_dir)
	with open(f'{sub_save_dir}{args.saveNm}.json', 'w') as outFile:
		json.dump(allData, outFile)
	
