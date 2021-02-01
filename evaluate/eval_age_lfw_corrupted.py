import keras
import sys, os, re
from glob import glob
from datetime import datetime
import time
from collections import defaultdict
import numpy as np

sys.path.append("../dataset")
sys.path.append("../training")
sys.path.append("../training/scratch_models")
sys.path.append('../training/keras_vggface/keras_vggface')

from dataset_tools import cut
from antialiasing import BlurPool
from mobile_net_v2_keras import relu6
from model_build import age_relu, Hswish, HSigmoid

custom_objects = {'BlurPool': BlurPool,
                  'relu6': relu6,
                  'age_relu': age_relu,
                  'Hswish': Hswish,
                  'HSigmoid': HSigmoid
                  }

from lfw_dataset_age import LFWPlusDatasetAge as Dataset
from evaluate_utils import load_keras_model, get_allchecks

import argparse

available_metrics = ["mean_squared_error", "mean_absolute_error", "mean_error"]

parser = argparse.ArgumentParser(description='LFW+ test.')
parser.add_argument('--path', dest='inpath', type=str, help='source path of model to test')
parser.add_argument('--gpu', dest="gpu", type=str, default="0", help="gpu to use")
parser.add_argument('--out_path', dest="outf", type=str, default="results_lfw", help='Path of results file')
parser.add_argument('--time', action='store_true')
parser.add_argument('--inference', action='store_true')
parser.add_argument('--nocorruption', action='store_true')
parser.add_argument('--metric', dest="metric", type=str, default="mean_squared_error", choices=available_metrics, help="gpu to use")
args = parser.parse_args()

CORRUPTED_LFW_PLUS_BLOB = '../dataset/corrupted_lfw_plus_dataset/lfw_plus.all.*'
CORRUPTED_CACHE_DIR = "../dataset/dataset_cache/corrupted_lfw_plus_gender_cache/{augmentation}"
LFW_PLUS_ORIGINAL = '../dataset/gender-access/lfw_cropped'
LFW_CSVMETA = '../dataset/gender-access/lfw_theirs.csv'

error_metric = None

def custom_mean_error(y_true, y_pred):
    return keras.backend.mean(y_pred-y_true)#, axis=-1)


# def classes_custom_mean_error(y_true, y_pred):
#     classes_error = defaultdict(list)
#     for y_true_instance, y_pred_instance in zip(y_true, y_pred):
#         classes_error[int(np.round(y_true_instance))].append(y_pred_instance - y_true_instance)
#     mean_error = [sum(classes_error[i])/len(classes_error[i]) if classes_error[i] else 0 for i in range(1, 101)]
#     return mean_error



def run_test(filepath,
            batch_size=64,
            dataset_dirs=LFW_PLUS_ORIGINAL,
            outf_path='results.txt',
            partition='test',
            inference=None
            ):

    print('Partition: %s' % partition)
    outf = open(outf_path, "a+")
    outf.write('Results for: %s\n' % filepath)
    if inference is not None:
        outf_inference = open(inference, "a+")
        outf_inference.write('Results for: %s\n' % filepath)
    model, INPUT_SHAPE = load_keras_model(filepath)

    batch_size = batch_size if inference is None else 1

    for d in glob(dataset_dirs):
        if dataset_dirs == LFW_PLUS_ORIGINAL:
            cache_dir = None
        else:
            augmentation = os.path.split(d)[-1]
            cache_dir = CORRUPTED_CACHE_DIR.format(augmentation=augmentation)
        dataset_test = Dataset(partition=partition,
                               target_shape=INPUT_SHAPE,
                               preprocessing='vggface2',
                               custom_augmentation=None,
                               augment=False,
                               imagesdir=d,
                               csvmeta=LFW_CSVMETA,
                               cache_dir=cache_dir)
        data_gen = dataset_test.get_generator(batch_size)
        print("Dataset batches %d" % len(data_gen))
        start_time = time.time()
        loss = keras.losses.mean_squared_error
        if error_metric == "mean_error":
            metrics = [custom_mean_error]
        elif error_metric == "mean_squared_error":
            metrics = [keras.metrics.mean_squared_error]
        elif error_metric == "mean_absolute_error":
            metrics = [keras.metrics.mean_absolute_error]
        else:
            raise Exception("{} is not available as metric".format(error_metric))
        model.compile(loss=loss, optimizer="sgd", metrics=metrics)
        result = model.evaluate_generator(data_gen, verbose=1, workers=4)
        spent_time = time.time() - start_time
        batch_average_time = spent_time / len(data_gen)
        print("Evaluate time %d s" % spent_time)
        print("Batch time %.10f s" % batch_average_time)
        o = "%s %f\n" % (d, result[1])
        print("\n\n RES " + o)
        outf.write(o)
        if inference is not None:
            outf_inference.write("%s %f ms\n" % (d, batch_average_time * 1000))
    outf.write('\n\n')
    outf.close()
    if inference is not None:
        outf_inference.write('\n\n')
        outf_inference.close()


def run_all(dirpath, outf_path, dataset_dirs, partition='test', inference=None):
    allchecks = get_allchecks(dirpath)
    for c in allchecks:
        print('\n Testing %s now...\n' % c)
        if inference is None:
            run_test(c, outf_path=outf_path, dataset_dirs=dataset_dirs, partition=partition)
        else:
            print("Warning Inference active")
            run_test(c, outf_path=outf_path, dataset_dirs=dataset_dirs, inference=inference, batch_size=1,
                     partition=partition)




if '__main__' == __name__:
    start_time = datetime.today()
    error_metric = args.metric
    os.makedirs(args.outf, exist_ok=True)

    if args.nocorruption:
        out_path = os.path.join(args.outf, "uncorrupted_results.txt")
    else:
        out_path = os.path.join(args.outf, "corrupted_results.txt")

    if args.time:
        out_path = "%s_%s%s" % (
            os.path.splitext(out_path)[0], start_time.strftime('%Y%m%d_%H%M%S'), os.path.splitext(out_path)[1])
        print("Exporting to %s" % out_path)

    if args.gpu is not None:
        gpu_to_use = [str(s) for s in args.gpu.split(',') if s.isdigit()]
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpu_to_use)
        print("WARNING: Using GPU %s" % os.environ["CUDA_VISIBLE_DEVICES"])

    imagesdir_set = LFW_PLUS_ORIGINAL if args.nocorruption else CORRUPTED_LFW_PLUS_BLOB
    print("test_set %s" % imagesdir_set)
    if args.inference:
        print("Warning Inference test activated")
        inference = "%s_%s%s" % (os.path.splitext(out_path)[0], 'inference', os.path.splitext(out_path)[1])
    else:
        inference = None
    if args.inpath.endswith('.hdf5'):
        print("Start single test mode")
        run_test(args.inpath, outf_path=out_path, dataset_dirs=imagesdir_set, inference=inference)
    else:
        run_all(args.inpath, outf_path=out_path, dataset_dirs=imagesdir_set, inference=inference)

    if args.gpu is not None and args.inference:
        start_time = datetime.today()
        os.environ["CUDA_VISIBLE_DEVICES"] = ''
        print("CPU mode activated")
        inference = "%s_%s%s" % (os.path.splitext(inference)[0], 'CPU', os.path.splitext(inference)[1])
        if args.inpath.endswith('.hdf5'):
            run_test(args.inpath, outf_path=out_path, dataset_dirs=imagesdir_set, inference=inference)
        else:
            run_all(args.inpath, outf_path=out_path, dataset_dirs=imagesdir_set, inference=inference)
        print("CPU execution time: %s" % str(datetime.today() - start_time))

    print("GPU execution time: %s" % str(datetime.today() - start_time))
