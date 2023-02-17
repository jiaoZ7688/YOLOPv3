<div align="center">
<h1> YOLOPv3: Better Multi-Task learning Network for Panoptic Driving Perception </h1>
</font></span> -->

  Jiao Zhan, Chi Guo, Yarong Luo, Jianlang Hu, Fei Li, Jingnan Liu
</div>

## News
* `2023-2-17`:  We've uploaded the experiment results along with some code, and the full code will be released soon!

## Introduction

Panoptic driving perception plays a key role in autonomous driving. To effectively solve this problem, existing methods generally adopt high-precision and real-time multi-task learning networks to perform multiple related tasks simultaneously. However, the performance and the training efficiency of these networks can hinder their practical deployment of networks. In this paper, we present vital improvements to the existing YOLOP, forming an efficient multi-task learning network that can simultaneously per-form traffic object detection, drivable area segmentation, and lane detection, named YOLOPv3. In terms of architecture improve-ments, we design an efficient network architecture to achieve a balance between accuracy and computation cost. In terms of net-work training, we propose an efficient training strategy to optimize the training process without additional inference cost. Our method not only improves the network performance, making it significantly better than existing methods, but also improves the training efficiency, making it more accessible to users with limited computing resources. Experimental results on the challenging BDD100K dataset demonstrate the state-of-the-art (SOTA) performance in real-time: It achieves 96.9 % recall and 84.3% mAP50 on traffic object detection, 93.2% mIoU on drivable area segmentation, and 88.3% accuracy and 28.0% IoU on lane detection. Meanwhile, it owns competitive inference speed compared to the lightweight network YOLOP. Thus, YOLOPv3 is an efficient solution for panoptic driving perception problems.

## Results
We used the BDD100K as our datasets,and experiments are run on **NVIDIA TESLA V100**.

### video visualization Results
model : trained on the BDD100k train set and test on the BDD100k val set .
<td><img src=demo/2.gif/></td>

### image visualization Results
model : trained on the BDD100k train set and test on the BDD100k val set .
<div align = 'None'>
  <img src="demo/all.jpg" width="100%" />
</div>


### Model parameter and inference speed
We compare YOLOPv3 with YOLOP and HybridNets on the NVIDIA RTX 3080. MRP denotes model re-parameterization techniques.

|        Model         |   Backbone   |   Params   | Speed (fps) |
|:--------------------:|:------------:|:----------:|:-----------:|
|       `YOLOP`        |  CSPDarknet  |    7.9M    |     39      |
|     `HybridNets`     | EfficientNet |    12.8M   |     17      |
|  `YOLOPv3 (no MRP)`  |    ELANNet   |    30.9M   |     26      |
|   `YOLOPv3 (MRP)`    |    ELANNet   |    30.2M   |     37      |


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


## Pre-trained Model
train log in `train.log`. we choose epoch-189.pth as final result.
You can get the pre-trained model from <a href="https://pan.baidu.com/s/1FlD9TtDdg6BuD6CSAu55ag">here</a>.
Extraction code：7a0c


## Dataset
For BDD100K: [imgs](https://bdd-data.berkeley.edu/), [det_annot](https://drive.google.com/file/d/1d5osZ83rLwda7mfT3zdgljDiQO3f9B5M/view), [da_seg_annot](https://drive.google.com/file/d/1yNYLtZ5GVscx7RzpOd8hS7Mh7Rs6l3Z3/view), [ll_seg_annot](https://drive.google.com/file/d/1BPsyAjikEM9fqsVNMIygvdVVPrmK1ot-/view)


## Demo Test

You can use the image or video as input.

```shell
python demo.py  --source demo/example.jpg
```

## License

YOLOPv3 is released under the [MIT Licence](LICENSE).

