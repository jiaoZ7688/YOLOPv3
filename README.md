<div align="center">
<h1> YOLOPv3: Better Multi-Task learning Network for Panoptic Driving Perception </h1>

[![Generic badge](https://img.shields.io/badge/License-MIT-<COLOR>.svg?style=for-the-badge)](https://github.com/jiaoZ7688/YOLOPv3/blob/main/LICENSE) 
[![PyTorch - Version](https://img.shields.io/badge/PYTORCH-1.12+-red?style=for-the-badge&logo=pytorch)](https://pytorch.org/get-started/locally/) 
[![Python - Version](https://img.shields.io/badge/PYTHON-3.7+-red?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
<br>

  Jiao Zhan, Chi Guo, Yarong Luo, Jianlang Hu, Fei Li, Jingnan Liu
</div>

## News
* `2023-2-17`:  We've uploaded the experiment results along with some code, and the full code will be released soon!

* `2023-8-26`:  We have uploaded part of the code and the full code will be released soon!

## Introduction

Panoptic driving perception is crucial for autonomous driving, encompassing traffic object detection, drivable area segmentation, and lane detection. Existing methods typically employ high-precision and real-time multi-task learning networks to tackle these tasks simultaneously. While they yield promising results, better performance can be achieved by resolving current problems such as suboptimal network structures and poor training efficiency. In this paper, we propose YOLOPv3, a simple yet efficient mul-ti-task learning network for panoptic driving perception. Compared to previous works, we make vital improvements. In terms of structure improvement, we design an excellent network structure to capture multi-scale high-resolution features and long-distance contextual dependencies, resulting in improved prediction performance. In terms of efficiency improvement, we propose an effi-cient training strategy to optimize the training process without additional inference cost, allowing our multi-task learning network to achieve optimal performance through simple end-to-end training. Experimental results on the challenging BDD100K dataset demonstrate the state-of-the-art (SOTA) performance of YOLOPv3: it achieves 96.9 % recall and 84.3% mAP50 on traffic object detection, 93.2% mIoU on drivable area segmentation, and 88.3% accuracy and 28.0% IoU on lane detection. Moreover, YOLOPv3 owns competitive inference speed compared to the lightweight network YOLOP. Therefore, YOLOPv3 is a powerful solution for panoptic driving perception problems.

## Results
* We used the BDD100K as our datasets,and experiments are run on **NVIDIA TESLA V100**.
* model : trained on the BDD100k train set and test on the BDD100k val set .

### video visualization Results
* Note: The raw video comes from [HybridNets](https://github.com/datvuthanh/HybridNets/tree/main/demo/video/)
* The results of our experiments are as follows:
<td><img src=demo/2.gif/></td>

### image visualization Results
* The results on the BDD100k val set.
<div align = 'None'>
  <img src="demo/all.jpg" width="100%" />
</div>


### Model parameter and inference speed
We compare YOLOPv3 with YOLOP and HybridNets on the NVIDIA RTX 3080. 
In terms of real-time, we compare the inference speed (excluding data pre-processing and NMS operations) at batch size 1.  
MRP denotes model re-parameterization techniques.


|        Model         |   Backbone   |   Params   | Flops | Speed (fps) |
|:--------------------:|:------------:|:----------:|:-----------:|:-----------:|
|       `YOLOP`        |  CSPDarknet  |    7.9M    |    9.3G     |     39      |
|     `HybridNets`     | EfficientNet |    12.8M   |    7.8G    |     17      |
|  `YOLOPv3 (no MRP)`  |    ELAN-Net   |    30.9M   |    36.0G    |     26      |
|   `YOLOPv3 (MRP)`    |    ELAN-Net   |    30.2M   |    35.4G     |     37      |


### Traffic Object Detection Result
<table>
<tr><th>Result </th><th>Visualization</th></tr>
<tr><td>

|        Model       |  Recall (%)  |   mAP@0.5 (%)   |
|:------------------:|:------------:|:---------------:|
|     `MultiNet`     |     81.3     |       60.2      |
|      `DLT-Net`     |     89.4     |       68.4      |
|   `Faster R-CNN`   |     77.2     |       55.6      |
|      `YOLOv5s`     |     86.8     |       77.2      |
|       `YOLOP`      |     89.2     |       76.5      |
|    `HybridNets`    |     92.8     |       77.3      |
|     `YOLOPv2`      |     91.1     |       83.4      |
|   **`YOLOPv3`**    |   **96.9**   |     **84.3**    |

</td><td>

<img src="demo/det.jpg" width="100%" />

</td></tr> </table>


### Drivable Area Segmentation
<table>
<tr><th>Result </th><th>Visualization</th></tr>
<tr><td>

|       Model      | Drivable mIoU (%) |
|:----------------:|:-----------------:|
|    `MultiNet`    |        71.6       |
|     `DLT-Net`    |        71.3       |
|     `PSPNet`     |        89.6       |
|      `YOLOP`     |        91.5       |
|   `HybridNets`   |        90.5       |
|    `YOLOPv2`     |      **93.2**     |
|  **`YOLOPv3`**   |      **93.2**     |

</td><td>

<img src="demo/da.jpg" width="100%" />

</td></tr> </table>


### Lane Line Detection
<table>
<tr><th>Result </th><th>Visualization</th></tr>
<tr><td>

|      Model       | Accuracy (%) | Lane Line IoU (%) |
|:----------------:|:------------:|:-----------------:|
|      `Enet`      |     34.12    |       14.64       |
|      `SCNN`      |     35.79    |       15.84       |
|    `Enet-SAD`    |     36.56    |       16.02       |
|      `YOLOP`     |     70.5     |       26.2        |
|    `HybridNets`  |     85.4     |     **31.6**      |
|     `YOLOPv2`    |     87.3     |       27.2        |
|   **`YOLOPv3`**  |   **88.3**   |       28.0        |

</td><td>

<img src="demo/ll.jpg" width="100%" />

</td></tr> </table>


## Project Structure

```python
├─inference
│ ├─image   # inference images
│ ├─image_output   # inference result
├─lib
│ ├─config/default   # configuration of training and validation
│ ├─core    
│ │ ├─activations.py   # activation function
│ │ ├─evaluate.py   # calculation of metric
│ │ ├─function.py   # training and validation of model
│ │ ├─general.py   #calculation of metric、nms、conversion of data-format、visualization
│ │ ├─loss.py   # loss function
│ │ ├─yolov7_loss.py   # yolov7's loss function
│ │ ├─yolov7_general.py   # yolov7's general function
│ │ ├─postprocess.py   # postprocess(refine da-seg and ll-seg, unrelated to paper)
│ ├─dataset
│ │ ├─AutoDriveDataset.py   # Superclass dataset，general function
│ │ ├─bdd.py   # Subclass dataset，specific function
│ │ ├─convect.py 
│ │ ├─DemoDataset.py   # demo dataset(image, video and stream)
│ ├─models
│ │ ├─YOLOP.py    # Setup and Configuration of model
│ │ ├─commom.py   # calculation module
│ ├─utils
│ │ ├─augmentations.py    # data augumentation
│ │ ├─autoanchor.py   # auto anchor(k-means)
│ │ ├─split_dataset.py  # (Campus scene, unrelated to paper)
│ │ ├─plot.py  # plot_box_and_mask
│ │ ├─utils.py  # logging、device_select、time_measure、optimizer_select、model_save&initialize 、Distributed training
│ ├─run
│ │ ├─dataset/training time  # Visualization, logging and model_save
├─tools
│ │ ├─demo.py    # demo(folder、camera)
│ │ ├─test.py    
│ │ ├─train.py    
├─weights    # Pretraining model
```

---

## Requirement

This codebase has been developed with python version 3.7, PyTorch 1.12+ and torchvision 0.13+
```setup
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```
See `requirements.txt` for additional dependencies and version requirements.
```setup
pip install -r requirements.txt
```

## Pre-trained Model
You can get the pre-trained model from <a href="https://pan.baidu.com/s/19wj4XOHReY8sGgCh787mOw">here</a>.
Extraction code：jbty


## Dataset
For BDD100K: [imgs](https://bdd-data.berkeley.edu/), [det_annot](https://drive.google.com/file/d/1d5osZ83rLwda7mfT3zdgljDiQO3f9B5M/view), [da_seg_annot](https://drive.google.com/file/d/1yNYLtZ5GVscx7RzpOd8hS7Mh7Rs6l3Z3/view), [ll_seg_annot](https://drive.google.com/file/d/1BPsyAjikEM9fqsVNMIygvdVVPrmK1ot-/view)

We recommend the dataset directory structure to be the following:

```
# The id represent the correspondence relation
├─dataset root
│ ├─images
│ │ ├─train
│ │ ├─val
│ ├─det_annotations
│ │ ├─train
│ │ ├─val
│ ├─da_seg_annotations
│ │ ├─train
│ │ ├─val
│ ├─ll_seg_annotations
│ │ ├─train
│ │ ├─val
```

Update the your dataset path in the `./lib/config/default.py`.

## Training
coming soon......

## Evaluation

```shell
python tools/test.py --weights weights/epoch-189.pth
```

## Demo

You can store the image or video in `--source`, and then save the reasoning result to `--save-dir`

```shell
python tools/demo.py --weights weights/epoch-189.pth
                     --source inference/image
                     --save-dir inference/image_output
                     --conf-thres 0.3
                     --iou-thres 0.45
```

## License

YOLOPv3 is released under the [MIT Licence](LICENSE).

## Acknowledgements

Our work would not be complete without the wonderful work of the following authors:

* [YOLOP](https://github.com/hustvl/YOLOP)
* [YOLOv5](https://github.com/ultralytics/yolov5)
* [YOLOv7](https://github.com/WongKinYiu/yolov7)
* [HybridNets](https://github.com/datvuthanh/HybridNets)
