import asyncio
from argparse import ArgumentParser

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)

import matplotlib
import matplotlib.pyplot as plt
import os
import cv2
matplotlib.use('Agg')
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--img', default='/root/autodl-tmp/coco/images/val2017/000000067213.jpg',help='Image file')
    parser.add_argument('--config', default='/root/autodl-tmp/openset_detection-main/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_cocoOS_Anchor.py', help='Config file')
    parser.add_argument('--checkpoint', default='/root/autodl-tmp/openset_detection-main/mmdetection/weights/frcnnCEwAnchorCocoOS-80/latest.pth', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


def main(args):
    #os.system('export DISPLAY=:0.0')
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_detector(model, args.img)
    print("finish")
    savepath = "test.jpg"
    #out_file = os.path.join(savepath, "test.jpg")
    # show the results
    img = show_result_pyplot(model, args.img, result, score_thr=args.score_thr)
    #show_result(imgout_file, result, model.CLASSES, out_file='result.jpg')
    print("finish")
    #cv2.imshow('imshow',img)
    cv2.imencode('.jpg', img)[1].tofile("/root/autodl-tmp/openset_detection-main/mmdetection/demo/tests.jpg")


async def async_main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)
    # show the results
    show_result_pyplot(model, args.img, result[0], score_thr=args.score_thr)


if __name__ == '__main__':
    args = parse_args()
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)
