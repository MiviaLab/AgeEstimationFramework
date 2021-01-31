# MIVIA Efficient Age Estimation Deep Learning Framework

A detailed explaination of the material contained in this repository is in the paper
```
Effective training of convolutional neural networks for age estimation based on knowledge distillation
A. Greco, A. Saggese, M. Vento, V. Vigilante
Neural Computing and Applications 2021.
```
If you find any of this content useful for your research, please cite the paper above.


## Content of this repository
- [X] Age annotation for the [VGGFace2 dataset](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/). [Download the annotation here.](https://github.com/MiviaLab/AgeEstimationFramework/releases/tag/0)
- [ ] Trained efficient deep CNN models for age from our paper **(coming soon)**
- [ ] Code used for training and evaluating the models on various datasets **(coming soon)**



## Abstract

Age  estimation  from  face  images  can  be profitably  employed  in  several  applications,  ranging from  digital  signage  to  social  robotics,  from  business intelligence to access control. Only in recent years, the advent  of  deep  learning  allowed  for  the  design  of  extremely accurate methods based on convolutional neural  networks  (CNNs)  that  achieve  a  remarkable  performance in various face analysis tasks.
However, these networks are not always applicable in real scenarios, due to  both  time  and  resource  constraints  that  the  most accurate  approaches  often  do  not  meet.  Moreover,  in case of age estimation, there is the lack of a large and reliably annotated dataset for training deep neural networks.

Within this context, we propose in this paper an effective training procedure of CNNs for age estimation based on knowledge distillation, able to allow smaller and simpler "student" models to be trained to match the predictions of a larger "teacher" model.
We experimentally  show  that  such  student  models  are  able  to almost reach the performance of the teacher, obtaining high accuracy over the LFW+, LAP 2016 and Adience datasets, but being up to 15 times faster. Furthermore, we evaluate the performance of the student models in presence of image corruptions, and we demonstrate that some of them are even more resilient to these corruptions than the teacher model.

