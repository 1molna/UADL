# UADL: Uncertainty-Driven Active Developmental Learning

This repository has the source code of the paper: 

**UADL: Uncertainty-Driven Active Developmental Learning**

Qinghua Hu, Luona Ji, Yu Wang, Shuai Zhaoa, Zhibin Lin

The proposed method is implemented based on the [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection).

![UADL](UADL.jpg)


**Abstract**

Existing machine learning models can well handle common classes but struggle to detect unfamiliar or unknown classes due to environmental variations. To address this challenge, we propose a new task called active developmental learning (ADL), which empowers models to actively determine what to learn in the open world, thereby progressively enhancing the capability of detecting unfamiliar and unknown classes. Considering the uncertain essence of the task, we design an uncertainty-driven method for ADL (UADL) that measures and utilizes uncertainty to evaluate unfamiliar known classes and unknown classes separately, which consists of two stages: (1) unfamiliar detection of known classes and (2) unknown detection of novel classes. In the first stage, UADL identifies unfamiliar samples of known classes via known-class uncertainty calculated by GMMs on detectors' heads. In the second stage, UADL identifies samples containing unknown classes via unknown-class uncertainty computed by class-specific GMMs in feature space. In both stages, uncertainty is used to select a minimal number of unlabeled samples for manual labeling, facilitating the model's active self-development. Experiments on multiple {object detection} benchmark datasets demonstrate the feasibility and significant performance of UADL and show its effectiveness against the ADL task compared to other state-of-the-art approaches.

## Acknowledgement 
The code is build on [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection) with some previous work code. If you use our code, please also consider cite:
```
@InProceedings{Choi_2021_ICCV,
    author    = {Choi, Jiwoong and Elezi, Ismail and Lee, Hyuk-Jae and Farabet, Clement and Alvarez, Jose M.},
    title     = {Active Learning for Deep Object Detection via Probabilistic Modeling},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {10264-10273}
}

@article{miller2021uncertainty,
  title={Uncertainty for Identifying Open-Set Errors in Visual Object Detection},
  author={Miller, Dimity and S{\"u}nderhauf, Niko and Milford, Michael and Dayoub, Feras},
  journal={IEEE Robotics and Automation Letters}, 
  year={2021},
  pages={1-1},
  doi={10.1109/LRA.2021.3123374}
}
```
