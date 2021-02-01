#!/usr/bin/python3
import warnings;
warnings.filterwarnings('ignore', category=FutureWarning)

import keras
import sys, os, re
from glob import glob
from datetime import datetime
import time

sys.path.append('emo')
from mobile_net_v2_keras import relu6

sys.path.append('emo/keras_vggface/keras_vggface')
from antialiasing import BlurPool

# from emo.keras_vggface.keras_vggface.antialiasing import BlurPool
# from emo.mobile_net_v2_keras import relu6

from feret_dataset_gender import FERETDatasetGender as Dataset

import argparse

parser = argparse.ArgumentParser(description='Feret gender test.')
parser.add_argument('--path', dest='inpath', type=str, help='source path of model to test')
parser.add_argument('--gpu', dest="gpu", type=str, default="0", help="gpu to use")
parser.add_argument('--outf', dest="outf", type=str, default=None, help='destination path of results file')
parser.add_argument('--time', action='store_true')
parser.add_argument('--inference', action='store_true')
parser.add_argument('--nocorruption', action='store_true')
args = parser.parse_args()

CORRUPTED_FERET_GENDER_BLOB = 'corrupted_feret_gender_dataset/feret_augmentation.*'
ORIGINAL_FERET_GENDER_BLOB = 'gender-feret'

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


def run_test(filepath,
            batch_size=64,
            dataset_dirs=ORIGINAL_FERET_GENDER_BLOB,
            outf_path='results.txt',
            partition='test',
            inference=None,
            extract_lambda=False
            ):

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

    for d in glob(dataset_dirs):
        if dataset_dirs == ORIGINAL_FERET_GENDER_BLOB:
            cache_dir = None
            detect_face = True
        else:
            detect_face = False
            augmentation = os.path.split(d)[-1]
            cache_dir = "corrupted_feret_cache/{augmentation}".format(augmentation=augmentation)
        dataset_test = Dataset(partition=partition,
                                target_shape=INPUT_SHAPE,
                                preprocessing='no_normalization',
                               custom_augmentation=None,
                               augment=False,
                               imagesdir=d,
                               cache_dir=cache_dir,
                               detect_face=detect_face)
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


def run_all(dirpath, outf_path, dataset_dirs, partition='test', inference=None, detect_face=True, cache_dir=None):
    allchecks = get_allchecks(dirpath)
    for c in allchecks:
        print('\n Testing %s now...\n' % c)
        if inference is None:
            run_test(c,
                     outf_path=outf_path,
                     dataset_dirs=dataset_dirs,
                     partition=partition
                     )
        else:
            print("Warning Inference active")
            run_test(c,
                    outf_path=outf_path,
                    dataset_dirs=dataset_dirs,
                    inference=inference,
                    batch_size=1,
                    partition=partition
                    )



if '__main__' == __name__:
    start_time = datetime.today()
    out_path = "results.txt" if args.outf is None else args.outf

    if args.time:
        out_path = "{}_{}{}".format(
            os.path.splitext(out_path)[0], 
            start_time.strftime('%Y%m%d_%H%M%S'), 
            os.path.splitext(out_path)[1])
        print("Exporting to %s" % out_path)

    if args.gpu is not None:
        gpu_to_use = [str(s) for s in args.gpu.split(',') if s.isdigit()]
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpu_to_use)
        print("WARNING: Using GPU %s" % os.environ["CUDA_VISIBLE_DEVICES"])

    imagesdir_set = ORIGINAL_FERET_GENDER_BLOB if args.nocorruption else CORRUPTED_FERET_GENDER_BLOB
    print("test_set", imagesdir_set)

    if args.inference:
        print("Warning Inference test activated")
        inference = "{}_{}{}".format(os.path.splitext(out_path)[0], 'inference', os.path.splitext(out_path)[1])
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
        inference = "{}_{}{}".format(os.path.splitext(inference)[0], 'CPU', os.path.splitext(inference)[1])
        if args.inpath.endswith('.hdf5'):
            run_test(args.inpath, outf_path=out_path, dataset_dirs=imagesdir_set, inference=inference)
        else:
            run_all(args.inpath, outf_path=out_path, dataset_dirs=imagesdir_set, inference=inference)
        print("CPU execution time:", str(datetime.today() - start_time))


    print("GPU execution time:", str(datetime.today() - start_time))
    exit(0)
