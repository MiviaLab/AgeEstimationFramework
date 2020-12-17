# AgeEstimationFramework
This repository will contain the code for *"Effective training of convolutional neural networks for age estimation based on knowledge distillation"*, under revision at "Neural Computing and Applications".

```diff
- Note: Code and models will be added after the acceptance of our eponymous work.
```

## Abstract

Age  estimation  from  face  images  can  beprofitably  employed  in  several  applications,  rangingfrom  digital  signage  to  social  robotics,  from  business intelligence to access control. Only in recent years, the advent  of  deep  learning  allowed  for  the  design  of  extremely accurate methods based on convolutional neural  networks  (CNNs)  that  achieve  a  remarkable  performance in various face analysis tasks.
However, these networks are not always applicable in real scenarios, due to  both  time  and  resource  constraints  that  the  most accurate  approaches  often  do  not  meet.  Moreover,  in case of age estimation, there is the lack of a large and reliably annotated dataset for training deep neural networks.

Within this context, we propose in this paper an effective training procedure of CNNs for age estimation based on knowledge distillation, able to allow smaller and simpler "student" models to be trained to match the predictions of a larger "teacher" model.
We experimentally  show  that  such  student  models  are  able  to almost reach the performance of the teacher, obtaining high accuracy over the LFW+, LAP 2016 and Adience datasets, but being up to 15 times faster. Furthermore,we evaluate the performance of the student models in presence of image corruptions, and we demonstrate that some of them are even more resilient to these corruptions than the teacher model.

