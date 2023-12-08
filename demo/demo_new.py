from argparse import ArgumentParser
import os
from mmdet.apis import inference_detector, init_detector  #, show_result_pyplot
import cv2

def show_result_pyplot(model, img, result, score_thr=0.3, fig_size=(15, 10)):
    """Visualize the detection results on the image.
    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        score_thr (float): The threshold to visualize the bboxes and masks.
        fig_size (tuple): Figure size of the pyplot figure.
    """
    if hasattr(model, 'module'):
        model = model.module
    img = model.show_result(img, result, score_thr=score_thr, show=False)
    return img
    # plt.figure(figsize=fig_size)
    # plt.imshow(mmcv.bgr2rgb(img))
    # plt.show()
 
 
def main():
    # config文件
    config_file = '/root/autodl-tmp/openset_detection-main/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    # 训练好的模型
    checkpoint_file = '/root/autodl-tmp/openset_detection-main/mmdetection/weights/frcnnCEwAnchorCocoOS-80/latest.pth'
    
    name = "/root/autodl-tmp/coco/images/val2017/000000208423.jpg"    
    # model = init_detector(config_file, checkpoint_file)
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

        # result = inference_detector(model, name)
        # model = init_detector(config_file, checkpoint_file, device='cuda:0')
    result = inference_detector(model, name)
    img = show_result_pyplot(model, name, result, score_thr=0.3)
    cv2.imwrite("/root/autodl-tmp/openset_detection-main/mmdetection/demo/208423.jpg", img)
 
 
if __name__ == '__main__':
    main()