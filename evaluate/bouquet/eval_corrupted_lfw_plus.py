#!/usr/bin/python3
import keras
import sys, os, re
from glob import glob
from datetime import datetime
import time

# TODO remove
sys.path.append('emo/keras_vggface/keras_vggface')
sys.path.append('emo')

from antialiasing import BlurPool
from mobile_net_v2_keras import relu6

import argparse

parser = argparse.ArgumentParser(description='LFW+ test.')
parser.add_argument('--path', dest='inpath', type=str, help='source path of model to test')
parser.add_argument('--gpu', dest="gpu", type=str, default="0", help="gpu to use")
parser.add_argument('--outf', dest="outf", type=str, default=None, help='destination path of results file')
parser.add_argument('--time', action='store_true')
parser.add_argument('--inference', action='store_true')
parser.add_argument('--nocorruption', action='store_true')
parser.add_argument('--dataset', dest='dataset', type=str, choices=['vggface2', 'lfw'], help="dataset on which to test")
args = parser.parse_args()

GENDER_ROOT = 'gender-access-2'
CORRUPTED_LFW_PLUS_BLOB = 'emo/corrupted_lfw_plus_dataset/lfw_plus.all.*'
LFW_PLUS_ORIGINAL = os.path.join(GENDER_ROOT, 'lfw_cropped')
PATTERN_CSV = 'lfw_theirs_<gender>.csv'

VGGFACE2_GENDER_ROOT = 'vggface2_data_2'


class InferenceCallbackTest(keras.callbacks.Callback):
    __slots__ = "start_predict_time", "start_batch_time", "batches_number", "out_file", "batches_time"

    def __init__(self, out_file):
        super().__init__()
        self.batches_number = 0
        self.out_file = out_file
        self.batches_time = list()
        self.start_predict_time = None
        self.start_batch_time = None

    def on_predict_begin(self, logs=None):
        print("Start measuring prediction")
        self.start_predict_time = time.time()

    def on_predict_end(self, logs=None):
        average = sum(self.batches_time) / len(self.batches_time)
        diff_time = time.time() - self.start_predict_time
        print('Predict {} predict duration: {}'.format(self.model.layers[1].name, diff_time))
        print('Average batch duration: {}'.format(average))
        with open(self.out_file, "a") as f:
            f.write("{}: {} ns".format(self.model.layers[1].name, average))

    def on_predict_batch_begin(self, batch, logs=None):
        self.start_batch_time = time.time()

    def on_predict_batch_end(self, batch, logs=None):
        self.batches_number += 1
        diff = time.time() - self.start_batch_time
        self.batches_time.append(diff)


def load_keras_model(filepath):
    model = keras.models.load_model(filepath, custom_objects={'BlurPool': BlurPool, 'relu6': relu6})
    if 'mobilenet96' in filepath:
        INPUT_SHAPE = (96, 96, 3)
    elif 'mobilenet64_bio' in filepath:
        INPUT_SHAPE = (64, 64, 3)
    elif 'xception71' in filepath:
        INPUT_SHAPE = (71, 71, 3)
    elif 'xception' in filepath:
        INPUT_SHAPE = (299, 299, 3)
    else:
        INPUT_SHAPE = (224, 224, 3)
    return model, INPUT_SHAPE


def run_test(filepath, batch_size=64, dataset_dirs=LFW_PLUS_ORIGINAL, outf_path='results.txt', partition='test',
             inference=None, extract_lambda=False):
    print('Partition: %s' % partition)
    outf = open(outf_path, "a+")
    outf.write('Results for: %s\n' % filepath)
    if inference is not None:
        outf_inference = open(inference, "a+")
        outf_inference.write('Results for: %s\n' % filepath)
    model, INPUT_SHAPE = load_keras_model(filepath)

    if extract_lambda:
        lambda_model = model.layers[-2]
        lambda_model.summary()
        model = keras.models.Model(lambda_model.layers[0].output, lambda_model.layers[-1].output)
        model.compile(optimizer=keras.optimizers.sgd(momentum=0.9), loss=keras.losses.categorical_crossentropy,
                  metrics=[keras.metrics.categorical_accuracy])
        model.summary()


    batch_size = batch_size if inference is None else 1
    # callbacks = [] if inference is None else [InferenceCallbackTest(inference)]
    # print(callbacks)

    for d in glob(dataset_dirs):
        dataset_test = Dataset(partition=partition, target_shape=INPUT_SHAPE, preprocessing='vggface2',
                               custom_augmentation=None, augment=False, imagesdir=d, cache_dir="lfw_cache",
                               csvmeta=os.path.join(GENDER_ROOT, PATTERN_CSV))
        data_gen = dataset_test.get_generator(batch_size)
        print("Dataset batches %d" % len(data_gen))
        start_time = time.time()
        result = model.evaluate_generator(data_gen, verbose=1, workers=4) #, callbacks=callbacks)
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


ep_re = re.compile('checkpoint.([0-9]+).hdf5')


def _find_latest_checkpoint(d):
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
    allchecks = [_find_latest_checkpoint(d) for d in alldirs]
    return [c for c in allchecks if c is not None]


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


def evalds(filepath, outf_path, partition, imagesdir=VGGFACE2_GENDER_ROOT, batch_size=64,
           change_root_cached=True):
    print('Partition: %s' % partition)
    outf = open(outf_path, "a+")
    outf.write('Results for: %s\n' % filepath)
    model, INPUT_SHAPE = load_keras_model(filepath)

    imagesdir += '/<part>'
    print("Dataset dir %s" % imagesdir.replace('<part>', partition))
    dataset_test = Dataset(partition, target_shape=INPUT_SHAPE, augment=False, imagesdir=imagesdir,
                           change_root_cached=change_root_cached, preprocessing='vggface2')

    data_gen = dataset_test.get_generator(batch_size)
    print("Dataset batches %d" % len(data_gen))
    start_time = time.time()
    result = model.evaluate_generator(data_gen, verbose=1, workers=4)
    spent_time = time.time() - start_time
    batch_average_time = spent_time / len(data_gen)
    print("Evaluate time %d s" % spent_time)
    print("Batch time %.10f s" % batch_average_time)
    o = "%s %f\n" % (partition, result[1])
    print("\n\n RES " + o)
    outf.write(o)

    outf.write('\n\n')
    outf.close()


def evalds_all(dirpath, outf_path, partition='test'):
    allchecks = get_allchecks(dirpath)
    for c in allchecks:
        print('\n Testing %s now...\n' % c)
        evalds(c, outf_path=outf_path, partition=partition)



if '__main__' == __name__:
    start_time = datetime.today()
    out_path = "results.txt" if args.outf is None else args.outf
    if args.time:
        out_path = "%s_%s%s" % (
            os.path.splitext(out_path)[0], start_time.strftime('%Y%m%d_%H%M%S'), os.path.splitext(out_path)[1])
        print("Exporting to %s" % out_path)

    if args.gpu is not None:
        gpu_to_use = [str(s) for s in args.gpu.split(',') if s.isdigit()]
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpu_to_use)
        print("WARNING: Using GPU %s" % os.environ["CUDA_VISIBLE_DEVICES"])

    if args.dataset.startswith('lfw'):
        from lfw_dataset_gender import LFWPlusDatasetGender as Dataset

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

    elif args.dataset.startswith('vggface2'):
        from vgg2_dataset_gender import Vgg2DatasetGender as Dataset
        partitions = ['train', 'val']
        for p in partitions:
            if args.inpath.endswith('.hdf5'):
                evalds(args.inpath, outf_path=out_path, partition=p)
            else:
                evalds_all(args.inpath, outf_path=out_path, partition=p)

    print("GPU execution time: %s" % str(datetime.today() - start_time))


# #### TEST CORRUPTION LFW ####
# python3 eval_corrupted_lfw_plus.py --gpu 0 --outf vggface_1_results.txt --path emo/out_training_fer/_netvgg16_datasetvggface2_gender_pretrainingvggface2_preprocessingvggface2_augmentationdefault_batch512_lr0.005_0.2_20_dataset_dirvggface2_data_2_sel_gpu0,1,2_ngpus3_training-epochs70_weight_decay0.005_momentum_20200311_141623/checkpoint.38.hdf5 --dataset lfw --time

# #### TEST NO CORRUPTION LFW ####
# python3 eval_corrupted_lfw_plus.py --gpu 1 --outf vggface_1_results_nocorr.txt --path emo/out_training_fer/_netvgg16_datasetvggface2_gender_pretrainingvggface2_preprocessingvggface2_augmentationdefault_batch512_lr0.005_0.2_20_dataset_dirvggface2_data_2_sel_gpu0,1,2_ngpus3_training-epochs70_weight_decay0.005_momentum_20200311_141623/checkpoint.43.hdf5 --dataset lfw --nocorruption --time

# #### TEST ALL CORRUPTION LFW ####
# python3 eval_corrupted_lfw_plus.py --gpu 3 --outf out_results/all_9_results.txt --path emo/out_training_fer --dataset lfw --time

# #### TEST ALL NO CORRUPTION LFW ####
# python3 eval_corrupted_lfw_plus.py --gpu 3 --outf out_results/all_9_results_nocorr.txt --path emo/out_training_fer --dataset lfw --nocorruption --time

# #### TEST ALL NO CORRUPTION VGGFACE2 ####
# python3 eval_corrupted_lfw_plus.py --gpu 1 --outf out_results_VGG2/all_9_vggface2_results_nocorr.txt --path emo/out_training_fer --dataset vggface2 --time

# #### INFERENCE TIME TEST ###
# python3 eval_corrupted_lfw_plus.py --gpu 3 --outf out_results_inference/all_9_results_nocorr.txt --path emo/out_training_fer --dataset lfw --nocorruption --time --inference
