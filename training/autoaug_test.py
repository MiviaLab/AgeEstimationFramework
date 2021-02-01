import sys, os
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import random

from rafdb_dataset import RAFDBDataset as Dataset, draw_emotion, NUM_CLASSES
#from autoaugment.augmentation_transforms import apply_policy
#from autoaugment.policies import good_policies

from autoaugment.rafdb_policies import rafdb_policies
from autoaugment.autoaug_transfs import apply_augment

def pil_wrap(img):
  if len(img.shape)==3 and img.shape[2]==1:
      img = np.squeeze(img,2)
  return Image.fromarray(img)

def pil_unwrap(pil_img):
  pic_array = np.array(pil_img.getdata()).reshape(pil_img.size[0], pil_img.size[1], -1)
  return pic_array.clip(0,255).astype(np.uint8)

def apply_policy(policy, img):
    pil_img = pil_wrap(img)
    for xform in policy:
        assert len(xform) == 3
        name, probability, level = xform
        pil_img = apply_augment(pil_img, name, level)
    pil_img = pil_img.convert('RGB')
    return pil_unwrap(pil_img)

g_chosen = None
class MyAutoAugmentation():
    def __init__(self, policies):
        self.policies = policies
    def before_cut(self, img, roi):
        return img
    def augment_roi(self, roi):
        return roi
    def after_cut(self, img):
        global g_chosen
        ##img = np.clip(img.astype(np.uint8), 0, 255)
        chosen = random.choice(self.policies)
        g_chosen = str(chosen)
        img = apply_policy(chosen, img)
        
        if len(img.shape)==2:
            img = np.expand_dims(img,2)
        #if img.shape[2]>1:
        #    img = img[:, :, ::-1]
        img = np.clip(img.astype(np.uint8), 0, 255)
        return img


def show_one_image():
    TARGET_SHAPE= (120,120,3)
    P = 'train'
    print('Partition: %s'%P)
    while True:
        NUM_ROWS = 6
        NUM_COLS = 10
        imout = np.zeros( (TARGET_SHAPE[0]*NUM_ROWS,TARGET_SHAPE[1]*NUM_COLS,3), dtype=np.uint8 )
        print(imout.shape)
        for ind1 in range(NUM_ROWS):
            for ind2 in range(NUM_COLS):
                a = MyAutoAugmentation(rafdb_policies)
                
                dataset_test = Dataset(partition=P, target_shape=TARGET_SHAPE,
                            debug_max_num_samples=1, augment=False, custom_augmentation=a)
                imex = np.squeeze(dataset_test.get_generator(1).__getitem__(0)[0],0)
                imex = ((imex*127)+127).clip(0,255).astype(np.uint8)
                cv2.putText(imex, g_chosen[:25], (0,10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), lineType=cv2.LINE_AA) 
                cv2.putText(imex, g_chosen[25:], (0,23), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), lineType=cv2.LINE_AA) 
                #imex_corrupted = a.before_cut(imex)
                imex_corrupted = imex
                off1=ind1*TARGET_SHAPE[0]
                off2=ind2*TARGET_SHAPE[1]
                imout[off1:off1+TARGET_SHAPE[0],off2:off2+TARGET_SHAPE[1],:] = imex_corrupted

        #imout = cv2.resize(imout, (TARGET_SHAPE[0]*2, TARGET_SHAPE[1]*2))
        cv2.imshow('imout', imout)
        k = cv2.waitKey(0)
        if k==27:
            sys.exit(0)


if '__main__' == __name__:
    show_one_image()
