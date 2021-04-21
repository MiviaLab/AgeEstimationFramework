# MIVIA Efficient Age Estimation Deep Learning Framework

A detailed explaination of the material contained in this repository is in [this paper](https://link.springer.com/article/10.1007/s00521-021-05981-0). 
If you find any of this content useful for your research, please cite
```
Effective training of convolutional neural networks for age estimation based on knowledge distillation
A. Greco, A. Saggese, M. Vento, V. Vigilante
Neural Computing and Applications 2021.
https://doi.org/10.1007/s00521-021-05981-0
```


For more details about the research in the field of age estimation, you can refer to: 
`
V. Carletti, A. Greco, G. Percannella and M. Vento, "Age from Faces in the Deep Learning Revolution," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 42, no. 9, pp. 2113-2132, 1 Sept. 2020, doi: 10.1109/TPAMI.2019.2910522.
`

## Content of this repository
- [X] Age annotation for the [VGGFace2 dataset](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/).
  - [Click to download annotation.](https://github.com/MiviaLab/AgeEstimationFramework/releases/tag/0)
- [X] **Trained efficient deep CNN models for age estimation** from our paper (Mobilenet, SE-ResNet50, DenseNet, VGG16).
  - [Click to download trained models](https://github.com/MiviaLab/AgeEstimationFramework/releases/tag/1)
- [X] Code used for training and evaluating the models on various datasets
  - [docs here](#Framework)



## Abstract

Age  estimation  from  face  images  can  be profitably  employed  in  several  applications,  ranging from  digital  signage  to  social  robotics,  from  business intelligence to access control. Only in recent years, the advent  of  deep  learning  allowed  for  the  design  of  extremely accurate methods based on convolutional neural  networks  (CNNs)  that  achieve  a  remarkable  performance in various face analysis tasks.
However, these networks are not always applicable in real scenarios, due to  both  time  and  resource  constraints  that  the  most accurate  approaches  often  do  not  meet.  Moreover,  in case of age estimation, there is the lack of a large and reliably annotated dataset for training deep neural networks.

Within this context, we propose in this paper an effective training procedure of CNNs for age estimation based on knowledge distillation, able to allow smaller and simpler "student" models to be trained to match the predictions of a larger "teacher" model.
We experimentally  show  that  such  student  models  are  able  to almost reach the performance of the teacher, obtaining high accuracy over the LFW+, LAP 2016 and Adience datasets, but being up to 15 times faster. Furthermore, we evaluate the performance of the student models in presence of image corruptions, and we demonstrate that some of them are even more resilient to these corruptions than the teacher model.


# Framework

## Setup
```
pip3 install -r requirements.txt
```

Images and annotation for the different datasets are not provided and need to be downloaded separately from the original sources.

## Dataset
The implemented _dataset_ are based on VGGFACE2, IMDB-WIKI (cleaned), LFW+, CHALEARN APPA-REAL and ADIENCE. <br>
Run these commands from dataset directory in order to verify that the images and annotations are correctly loaded.

```bash
python3 vggface2_dataset_age.py
python3 imdbwiki_dataset_age.py
python3 lfw_dataset_age.py
python3 chalearn_lap_appa_real_age.py
python3 adience_dataset_age.py
```

Images and annotation are not provided and need to be downloaded separately. Annotation for VMAGE (MIVIA VGGFace2 Age annotation) is [provided here](https://github.com/MiviaLab/AgeEstimationFramework/releases/tag/0).

## Train
In order to train neural networks, you must run <code>train.py</code> script from the _training_ directory.<br>
Here the used commands to train the associated paper solutions.

```bash
python3 train.py --net mobilenet224 --dataset vggface2_age --pretraining imagenet --preprocessing vggface2 --augmentation default --batch 256 --lr 0.005:0.2:20  --sel_gpu 0 --training-epochs 70
python3 train.py --net mobilenet224 --dataset imdbwiki_age --pretraining imagenet --preprocessing vggface2 --augmentation default --batch 256 --lr 0.005:0.2:20  --sel_gpu 0 --training-epochs 70
```
```bash
python3 train.py --net mobilenet96 --dataset vggface2_age --pretraining imagenet --preprocessing vggface2 --augmentation default --batch 256 --lr 0.005:0.2:20  --sel_gpu 0 --training-epochs 70
python3 train.py --net mobilenet96 --dataset imdbwiki_age --pretraining imagenet --preprocessing vggface2 --augmentation default --batch 256 --lr 0.005:0.2:20  --sel_gpu 0 --training-epochs 70
```
```bash
python3 train.py --net vgg16 --dataset vggface2_age --pretraining imagenet --preprocessing vggface2 --augmentation default --batch 128 --lr 0.005:0.2:20  --sel_gpu 0 --training-epochs 70
python3 train.py --net vgg16 --dataset imdbwiki_age --pretraining imagenet --preprocessing vggface2 --augmentation default --batch 128 --lr 0.005:0.2:20  --sel_gpu 0 --training-epochs 70
```
```bash
python3 train.py --net senet50 --dataset vggface2_age --pretraining imagenet --preprocessing vggface2 --augmentation default --batch 128 --lr 0.005:0.2:20  --sel_gpu 0 --training-epochs 70
python3 train.py --net senet50 --dataset imdbwiki_age --pretraining imagenet --preprocessing vggface2 --augmentation default --batch 128 --lr 0.005:0.2:20  --sel_gpu 0 --training-epochs 70
```
```bash
python3 train.py --net densenet121bc --dataset vggface2_age --pretraining imagenet --preprocessing vggface2 --augmentation default --batch 128 --lr 0.005:0.2:20  --sel_gpu 0 --training-epochs 70
python3 train.py --net densenet121bc --dataset imdbwiki_age --pretraining imagenet --preprocessing vggface2 --augmentation default --batch 128 --lr 0.005:0.2:20  --sel_gpu 0 --training-epochs 70
```
```bash
python3 train.py --net mobilenetv3 --dataset vggface2_age --pretraining imagenet --preprocessing vggface2 --augmentation default --batch 128 --lr 0.005:0.2:20  --sel_gpu 0 --training-epochs 70
python3 train.py --net mobilenetv3 --dataset imdbwiki_age --pretraining imagenet --preprocessing vggface2 --augmentation default --batch 128 --lr 0.005:0.2:20  --sel_gpu 0 --training-epochs 70
```
```bash
python3 train.py --net mobilenetv3small --dataset vggface2_age --pretraining imagenet --preprocessing vggface2 --augmentation default --batch 128 --lr 0.005:0.2:20  --sel_gpu 0 --training-epochs 70
python3 train.py --net mobilenetv3small --dataset imdbwiki_age --pretraining imagenet --preprocessing vggface2 --augmentation default --batch 128 --lr 0.005:0.2:20  --sel_gpu 0 --training-epochs 70
```
```bash
python3 train.py --net mobilenetv3large --dataset vggface2_age --pretraining imagenet --preprocessing vggface2 --augmentation default --batch 128 --lr 0.005:0.2:20  --sel_gpu 0 --training-epochs 70
python3 train.py --net mobilenetv3large --dataset imdbwiki_age --pretraining imagenet --preprocessing vggface2 --augmentation default --batch 128 --lr 0.005:0.2:20  --sel_gpu 0 --training-epochs 70
```

Pretrained models can be [downloaded here](https://github.com/MiviaLab/AgeEstimationFramework/releases/tag/1).

## Fine-tuning
We provide an interface in order to perform fine-tuning over CHALEARN APPA-REAL dataset (train and validation set) and over ADIENCE dataset (5 folds: 3 train and 1 validation, leaving 1 fold for testing purpose, used in rotation, e.g. train:0-1-2 val:3, train:1-2-3 val:4, etc.).
You can change the fine-tuning parameters directly in the interface code.
In the example commands we use SENet-50 network directory.

```bash
python3 train_chalearn_interface.py ../trained_ended/_netsenet50_datasetvggface2_age_pretrainingimagenet_preprocessingvggface2_augmentationdefault_batch128_lr0.005_0.2_20_sel_gpu2_training-epochs70_20200528_154836/
python3 train_chalearn_interface.py ../trained/_netmobilenet96_datasetvggface2_age_pretrainingimagenet_preprocessingvggface2_augmentationdefault_batch256_lr0.005_0.2_20_sel_gpu1_training-epochs70_20200613_004027/
```
```bash
python3 train_adience_interface.py ../trained_ended/_netsenet50_datasetvggface2_age_pretrainingimagenet_preprocessingvggface2_augmentationdefault_batch128_lr0.005_0.2_20_sel_gpu2_training-epochs70_20200528_154836/
python3 train_adience_interface.py ../trained_ended/_netmobilenet96_datasetvggface2_age_pretrainingimagenet_preprocessingvggface2_augmentationdefault_batch256_lr0.005_0.2_20_sel_gpu1_training-epochs70_20200613_004027/
```

Pretrained models can be [downloaded here](https://github.com/MiviaLab/AgeEstimationFramework/releases/tag/1)

## Evaluation
In order to evaluate the networks, move into the _evaluate_ directory and run the following commands according to the dataset you want to test which. In the subdirectory _results_, as the name suggests, you will find the results of these scripts, divided by dataset.

For each dataset, the provided commands must be executed in order beacuse each command depends on the results of the previous ones.


