# ID_CNN
Intrinsic Dimension of Convolutional Neural Networks

One of the geometric properties of representing data in neural network is Intrinsic Dimension i.e. minimum number of co-ordinates required to represent data without information loss. Local ID estimators compute in local subspaces of data representation. Global ID estimators compute over whole data point representation. Both global and local ID estimators can be used for estimation in alternate data neighbourhood. Our aim is to estimate ID at different layers for object detection networks and determine the relationship between average precision of augmented data and estimated ID. ID characteristics are distinguishable for normal and adversarial generated samples in local space. This motivates us to experiment with ID estimation in global space. TwoNN algorithm is implemented in our paper to estimate ID.

## Datasets

| Dataset | **KITTI** | **COCO** | **VOC** |
| -------- | --------- | --------- | --------- | 
| Number of Classes | 10 | 91 | 20 |
| Training samples | 5500 | 80000| 13500 |
| val samples | 1981 | 38000 | 3625 |
| Image Sizes | 1200 x 1200, 300 x 300 | 300 x 300 | 300 x 300 |

## Training and Evaluation

## Plots

![Image1](Plots/1.png)

The above plot is the median of Intrinsic dimension of all datasets on Faster RCNN with VGG-16 and VGG-19 backbones.
![Image2](Plots/2.png)
![Image3](Plots/3.png)

The plot on the left compares COCO and VOC data set on models trained on VOC and COCO data. We use alternate model for test data. The right plot shows median ID of KITTI data on RetinaNet with VGG-16 and VGG-19 backbones.

![Image4](Plots/4.png)
![Image5](Plots/5.png)
![Image6](Plots/6.png)

The plots i) KITTI  ii} VOC  iii) COCO are for Faster RCNN on VGG-16 for different augmentations of the data sets.

![Image7](Plots/Kitti.png)
![Image8](Plots/COCO.png)
![Image9](Plots/VOC.png)

Comparison of KITTI, COCO and VOC on Faster RCNN with VGG-16, VGG-19 and Res-101 backbones.

## Conclusions


## References
1. Intrinsic Dimension calculation - https://github.com/ansuini/IntrinsicDimDeep
2. KITTI dataset class - https://github.com/keshik6/KITTI-2d-object-detection
3. Image augmentations - https://towardsdatascience.com/complete-image-augmentation-in-opencv-31a6b02694f5
4. Fractal Dimension - https://github.com/ajaychawda58/Fractal-Dimension
