import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", dest="path", type=str, required=True, help="Path of nets")
parser.add_argument("--out_path", dest="out_path", type=str, default="results", help="Directory into which to store results") 
parser.add_argument("--batch_size", dest="batch_size", type=int, default=64, help="Batch size")
parser.add_argument('--gpu', dest="gpu", type=str, default=None, help="Gpu to use")
parser.add_argument("--avoid_roi", action="store_true", help="No roi will be focused")
args = parser.parse_args()

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
from cv2 import cv2
import csv
import numpy as np
from glob import glob
import re
import keras
import random
from tqdm import tqdm
import sys
from tabulate import tabulate
from collections import defaultdict
from datetime import datetime
from ast import literal_eval

sys.path.append("../dataset")
sys.path.append("../training")
sys.path.append("../training/scratch_models")
sys.path.append('../training/keras_vggface/keras_vggface')

from dataset_tools import cut
from antialiasing import BlurPool
from mobile_net_v2_keras import relu6
from model_build import age_relu, Hswish, HSigmoid

from evaluate_utils import *
from adience_dataset_age import AdienceAge as Dataset, CLASSES

CSV_FILE_NAME = "results_adience_{}_of{}.csv"
TABULATE_FILE_NAME = "tabulate_adience_of_{}_{}.txt"

FINETUNED_GLOB = "train_folds_{}_val_folds_{}"

FOLDS = ["fold_0", "fold_1", "fold_2", "fold_3", "fold_4"]
LATEX_TAB = False


def load_keras_model_adience(filepath):
    main_filepath = os.path.split(os.path.split(os.path.split(filepath)[0])[0])[1]
    print("Loading from:", main_filepath)
    if 'vgg16' in main_filepath:
        from train_adience import build_adience_finetuned
        model = vgg16_keras_build()[0]
        model = build_adience_finetuned(model)
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

def _refactor_data(data):
    restored = list()
    for item in data:
        truth = [literal_eval(item[2]), literal_eval(item[3])]
        roi = [np.int(x) for x in item[4:8]]
        item  = [item[0], literal_eval(item[1]), truth, roi]
        restored.append(item)
    return restored

def _refactor_data(data):
    restored = list()
    for item in data:
        path = item[0]
        pred = int(item[1])
        original = int(item[2])
        roi = [np.int(x) for x in item[3:7]]
        restored.append([path, pred, original, roi])
    return restored

def _percentage(item, total):
    # return 100 * item / total
    return item/total

# def _one_off(value):
#     # ask for 100+ age
#     for i in range(len(CLASSES)-1):
#         _, minimum = CLASSES[i]
#         maximum, _ = CLASSES[i+1]
#         if minimum < value < maximum:
#             return True
#     return False


# def _not_categorical_statistics(reference):
#     top_one_result = 0
#     one_off_result = 0
#     for _, predicted, original, _ in reference:
#         minimum, maximum = original[0], original[1]
#         if minimum <= predicted <= maximum:
#             top_one_result += 1
#             one_off_result += 1
#         elif _one_off(predicted):
#             one_off_result += 1
#     top_one_result = _percentage(top_one_result, len(reference))
#     one_off_result = _percentage(one_off_result, len(reference))
#     return top_one_result, one_off_result


def _categorical_statistics(reference):
    top_one_result = 0
    one_off_result = 0
    for _, predicted, original, _ in reference:
        if predicted == original:
            top_one_result += 1
        if original - 1 <= predicted <= original + 1:
            one_off_result += 1
    from sklearn.metrics import confusion_matrix
    allorig = [o for _,p,o,_ in reference]
    allpred = [p for _,p,o,_ in reference]
    cmat = confusion_matrix(allorig,allpred)
    print("Confusion matrix for fold")
    print(cmat)
    top_one_result = _percentage(top_one_result, len(reference))
    one_off_result = _percentage(one_off_result, len(reference))
    return top_one_result, one_off_result, cmat


def _infer_test_fold(path):
    absolute_dir = os.path.split(path)[0]
    only_fold_path = os.path.split(absolute_dir)[-1]
    print("Train and validation:", only_fold_path)
    infered_fold = list()
    for fold in FOLDS:
        if fold not in only_fold_path:
            infered_fold.append(fold)
    assert len(infered_fold) == 1, "Error in deserialization of the path: infered folds {}".format(infered_fold)
    print("Infered test fold:", infered_fold[0])
    return infered_fold[0]

    
# implement corruptions loop
def run_test(modelpath, batch_size=64, fold="fold_0"):
    model, INPUT_SHAPE = load_keras_model_adience(modelpath)
    dataset = Dataset(fold=fold, target_shape=INPUT_SHAPE, augment=False, preprocessing='vggface2')
    data_gen = dataset.get_generator(batch_size, fullinfo=True)
    original_labels = list()
    image_paths = list()
    image_rois = list()
    predictions = list()
    for batch in tqdm(data_gen):
        predictions.extend(np.argmax(model.predict(batch[0]), axis=1))
        original_labels.extend(np.argmax(batch[1], axis=1))
        image_paths.extend(batch[2])
        image_rois.extend(batch[3])
    assert (len(image_paths) == len(predictions) == len(original_labels) == len(image_rois)), "Invalid prediction on batch"
    return image_paths, predictions, original_labels, image_rois


def run_over_net(path, batch_size=64):
    all_results = defaultdict(dict)

    for p in get_allchecks(path):
        print("Testing", p)
        fold = _infer_test_fold(p)
        results_out_path = CSV_FILE_NAME.format(fold, p.split('/')[-3])
        if os.path.exists(os.path.join(args.out_path, results_out_path)):
            print("Open cached results")
            reference = _refactor_data(readcsv(os.path.join(args.out_path, results_out_path)))
        else:
            print("Running scratch test...")
            image_paths, predictions, original_labels, image_rois  = run_test(p, batch_size, fold)
            reference = zip_reference(image_paths, predictions, original_labels, image_rois)
            writecsv(os.path.join(args.out_path, results_out_path), reference)
        all_results[p][fold] = reference
    return all_results


def zip_reference(image_paths, predictions, original_labels, image_rois):
    reference = list()
    for path, pred, original, roi in zip(image_paths, predictions, original_labels, image_rois):
        reference.append((path, pred, original, roi))
    return reference


def _get_model_name(modeldirectory):
    return modeldirectory.split("_")[1][3:]

def _get_model_name_from_checkpoint_path(checkpoint_path):
    path = os.path.split(checkpoint_path)[0]
    path = os.path.split(path)[0]
    path = os.path.split(path)[1]
    return path.split("_")[3][3:]


def _tabulate_accuracy(outpath, data, main_label):
    col_labels = ['Method'] + [main_label + " " + f for f in FOLDS] + [main_label + " AVG"]
    # row_labels = [m for m in data.keys()]
    # lfw_values = [v for v in uncorr_data.values()]
    # lfw_c_values = [sum(corr_dict.values()) / len(corr_dict.values()) for corr_dict in data_means.values()]
    # table_vals = [[r, round(v, 3), round(c, 3)] for r, v, c in zip(row_labels, lfw_values, lfw_c_values)]

    # top_one_reference[_get_model_name(args.path.split('/')[-2])][fold] = top_one_result

    table_vals = list()

    compress_data = dict()
    for model_path, fold_data in data.items():
        for fold_key, fold_value in fold_data.items():
            if fold_key in compress_data:
                raise Exception("Not unique folds in testing.")
            compress_data[fold_key] = fold_value
    
    values = [compress_data[fold] for fold in FOLDS]
    avg = sum(values)/len(values)

    model_name = _get_model_name_from_checkpoint_path(model_path)
    table_vals.append([model_name] + values + [avg])

    tab_1 = tabulate(table_vals, headers=col_labels, tablefmt="latex" if LATEX_TAB else "grid")

    print(main_label)
    print(tab_1)
    with open(outpath, 'w') as f:
        f.write(tab_1)


if __name__ == '__main__':
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)

    if args.gpu is not None:
        select_gpu(args.gpu)

    top_one_reference = defaultdict(dict)
    one_off_reference = defaultdict(dict)

    top_one_label = "Top-1"
    one_off_label = "1-off"
    
    # if args.path.endswith('.hdf5'):
    #     for fold in FOLDS:
    #         results_out_path = CSV_FILE_NAME.format(fold, args.path.split('/')[-2])
    #         if os.path.exists(os.path.join(args.out_path, results_out_path)):
    #             reference = _refactor_data(readcsv(os.path.join(args.out_path, results_out_path)))
    #         else:
    #             image_paths, predictions, original_labels, image_rois  = run_test(args.path, args.batch_size, fold)
    #             reference = zip_reference(image_paths, predictions, original_labels, image_rois)
    #             writecsv(os.path.join(args.out_path, results_out_path), reference)         
            
    #         top_one_result, one_off_result = _categorical_statistics(reference)
    #         top_one_reference[_get_model_name(args.path.split('/')[-2])][fold] = top_one_result
    #         one_off_reference[_get_model_name(args.path.split('/')[-2])][fold] = one_off_result

    #     tabulate_path = os.path.join(args.out_path, TABULATE_FILE_NAME.format("top_one", args.path.split('/')[-2]))
    #     _tabulate_accuracy(tabulate_path, top_one_reference, top_one_label)

    #     tabulate_path = os.path.join(args.out_path, TABULATE_FILE_NAME.format("one_off", args.path.split('/')[-2]))
    #     _tabulate_accuracy(tabulate_path, one_off_reference, one_off_label)        

    # else:


    results = run_over_net(args.path)
    # all_results[p][fold] = reference
    cmat_all = []
    # print(results)
    for model_path, data_fold in results.items(): 
        for fold, reference in data_fold.items():
            top_one_result, one_off_result, cmat = _categorical_statistics(reference)
            top_one_reference[model_path][fold] = top_one_result
            one_off_reference[model_path][fold] = one_off_result
            cmat_all.append(cmat)
    cmat_all = np.sum(cmat_all, axis=0)
    print('Confusion matrix for all folds')
    print(cmat_all)

    tab_name = datetime.today().strftime('results_%Y%m%d_%H%M%S')
    

    tabulate_path = os.path.join(args.out_path, TABULATE_FILE_NAME.format("top_one", tab_name))
    _tabulate_accuracy(tabulate_path, top_one_reference, top_one_label)

    tabulate_path = os.path.join(args.out_path, TABULATE_FILE_NAME.format("one_off", tab_name))
    _tabulate_accuracy(tabulate_path, one_off_reference, one_off_label)
        

    


