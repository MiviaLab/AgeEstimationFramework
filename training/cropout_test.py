#!/usr/bin/python3
import tensorflow as tf
import keras
import os
import sys
from ferplus_dataset import FerPlusDataset, draw_emotion, NUM_CLASSES
from tqdm import tqdm
import cv2
import numpy as np
from dataset_tools import get_random_eraser


class CropoutAugmentation():
    def __init__(self):
        self.eraser = get_random_eraser()
    def before_cut(self, img):
        return img
    def after_cut(self, img):
        return self.eraser(img)



def show_one_image():
    TARGET_SHAPE= (48,48,3)
    P = 'PublicTest'
    print('Partition: %s'%P)
    while True:
        NUM_LEVELS = 10
        imout = np.zeros( (TARGET_SHAPE[0],TARGET_SHAPE[1]*NUM_LEVELS,3), dtype=np.uint8 )
        print(imout.shape)
        for ind1,ctypes in enumerate(['']):
            for ind2 in range(NUM_LEVELS):
                a = CropoutAugmentation()
                
                dataset_test = FerPlusDataset(partition=P, target_shape=TARGET_SHAPE,
                            debug_max_num_samples=1, augment=False, custom_augmentation=a)
                imex = np.squeeze(dataset_test.get_generator(1).__getitem__(0)[0],0)
                imex = ((imex*127)+127).clip(0,255).astype(np.uint8)
                
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
