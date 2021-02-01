import keras
import os
from cv2 import cv2
import csv
import numpy as np
from glob import glob
import re
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import  defaultdict
from tabulate import tabulate
import sys

# from keras.applications import keras_modules_injection

sys.path.append("../dataset")
sys.path.append("../training")
sys.path.append("../training/scratch_models")
sys.path.append('../training/keras_vggface/keras_vggface')

from dataset_tools import cut
from antialiasing import BlurPool
from mobile_net_v2_keras import relu6
from model_build import age_relu, Hswish, HSigmoid, vgg16_keras_build

custom_objects = {'BlurPool': BlurPool,
                  'relu6': relu6,
                  'age_relu': age_relu,
                  'Hswish': Hswish,
                  'HSigmoid': HSigmoid
                  }

def writecsv(path, data):
    with open(path,"w") as write_obj:
        writer = csv.writer(write_obj)
        for line in data:
            if line[3] is not None:
                extended_line = line[:3] + tuple(line[3])
                writer.writerow(extended_line)
            else:
                writer.writerow(line)


def readcsv(csvpath, debug_max_num_samples=None):
    data = list()
    with open(csvpath, newline='', encoding="utf8") as csvfile:
        reader = csv.reader(csvfile, skipinitialspace=True, delimiter=',', quotechar='|')
        for i, row in enumerate(reader):
            if debug_max_num_samples is not None and i >= debug_max_num_samples:
                break
            data.append(row)
    return np.array(data)


def select_gpu(gpu):
    assert type(gpu) is str, "Invalid GPU device. Parameter type not string"
    gpu_to_use = [str(s) for s in gpu.split(',') if s.isdigit()]
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpu_to_use)
    print("WARNING: Using GPU %s" % os.environ["CUDA_VISIBLE_DEVICES"])


def load_keras_model(filepath):
    main_filepath = os.path.split(os.path.split(filepath)[0])[1]
    print("Loading from:", main_filepath)
    if 'vgg16' in main_filepath:
        model = vgg16_keras_build()[0]
        model.load_weights(filepath)
        INPUT_SHAPE = (224, 224, 3)
    else:
        model = keras.models.load_model(filepath, custom_objects=custom_objects)
        if 'mobilenet96' in main_filepath:
            INPUT_SHAPE = (96, 96, 3)
        elif 'mobilenet64_bio' in main_filepath:
            INPUT_SHAPE = (64, 64, 3)
        elif 'xception71' in main_filepath:
            INPUT_SHAPE = (71, 71, 3)
        elif 'xception' in main_filepath:
            INPUT_SHAPE = (299, 299, 3)
        else:
            INPUT_SHAPE = (224, 224, 3)
    return model, INPUT_SHAPE


def find_latest_checkpoint(d):
    ep_re = re.compile('checkpoint.([0-9]+).hdf5')
    all_checks = glob(os.path.join(d, '*'))
    max_ep = 0
    max_c = None
    for c in all_checks:
        epoch_num = re.search(ep_re, c)
        if epoch_num is not None:
            epoch_num = int(epoch_num.groups(1)[0])
            if epoch_num > max_ep:
                max_ep = epoch_num
                max_c = c
    return max_c


def get_allchecks(dirpath):
    alldirs = glob(os.path.join(dirpath, '*'))
    allchecks = [find_latest_checkpoint(d) for d in alldirs]
    return [c for c in allchecks if c is not None]