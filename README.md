# Learnable-Path-Signature

Here is the official implementation of our work "***[Skeleton-based Gesture Recognition with Learnable Paths and Signature Features](https://ieeexplore.ieee.org/abstract/document/10261439)***". It is jointly accomplished by Yu Li, Honghui Lin, Dongzi Shi, and me. Chenyang Li initiated this project. Dr. Xin Zhang is the supervisor and provided lots of helpful suggestions.

In this work, we unprecedently combine the *path signature theory* with the *deep learning algorithms* on the task of gesture recognition. Specifically, we propose two innovative modules, i.e., the spatial-temporal path signature module (**ST-PSM**) and the learnable path signature module (**L-PSM**) to effectively describe the complex and dynamics cross spatiotemporal patterns in this challenging task. These modules are both plug-and-play which could be efficiently combined with the current graph convolution neural network and provide the complementary high-order differential information and thus further improve their performance. 

Extensive experiments were conducted on three different datasets for a comprehensive validation, i.e., ChaLearn2013, ChaLearn2016, and AUTSL.

This work is currently under review by the IEEE Transactions of Multimedia. We will release the detailed description of our work upon acceptance.

+++

### Implementation

We currently provide the implementation and the pre-trained models for ChaLearn2013 (Top1 accuracy: 94.18%) and ChaLearn2016 (Top1 accuracy: 51.60%) respectively.

Specifically,

+ *./ChaLearn2013* provides the codes for ChaLearn2013 dataset.
+ *./ChaLearn2016* provides the codes for ChaLearn2016 dataset.

For each dataset,

+ ***./src*** provides the implementation of our work and the other baselines. Within them,
  + ***./src/model/*** is the implementation of our proposed network,
  + ***the others*** is the official implementation of our compared methods.
+ ***./checkpoints*** provides the pre-trained network parameters.
+ ***./configs*** provides the network and training configurations corresponding to the pre-trained models in ./checkpoints .
+ ***./tools*** provides the detailed training, validation, and testing procedure we used.
+ ***./train.sh*** is the bash file for training. You can select different experiment configuration and train your own model by modifying this file.

Please contact us via eexinzhang@scut.edu.cn and jialecheng100@gmail.com if there is any problem or your can directly report the issues here. 

+++++++

### Reference

Please cite our work as follows if it is involved in your researches. Thank you!

***Cheng, J., Shi, D., Li, C., Li, Y., Ni, H., Jin, L., & Zhang, X. (2023). Skeleton-Based Gesture Recognition With Learnable Paths and Signature Features. IEEE Transactions on Multimedia.***