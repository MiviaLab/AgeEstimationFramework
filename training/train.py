import argparse

# available_nets = ['senet50', 'vgg16', 'densenet121bc', 'xception', 'xception71', 'mobilenet96', 'mobilenet224',
#                   'mobilenet64_bio', 'shufflenet224', 'squeezenet']

available_nets = ['senet50', 'vgg16', 'densenet121bc', 'mobilenet224', 'mobilenet96', 'mobilenetv3small', 'mobilenetv3large',
                'resnet50', 'mobilenetv3large_112', 'resnet50_112', 'senet50_112']

available_datasets = ['vggface2_age', 'imdbwiki_age']
available_normalizations = ['z_normalization', 'full_normalization', 'vggface2', "no_normalization"]
available_augmentations = ['default', 'vggface2', 'autoaugment-rafdb', 'no']
available_modes = ['train', 'training', 'test', 'train_inference', 'test_inference']
available_lpf = [0, 1, 2, 3, 5, 7]

parser = argparse.ArgumentParser(description='Common training and evaluation.')
parser.add_argument('--lpf', dest='lpf_size', type=int, choices=available_lpf, default=1, help='size of the lpf filter (1 means no filtering)')
parser.add_argument('--cutout', action='store_true', help='use cutout augmentation')
parser.add_argument('--center_loss', action='store_true', help='use center loss')
parser.add_argument('--weight_decay', type=float)
parser.add_argument('--lr', default='0.002', help='Initial learning rate or init:factor:epochs', type=str)
parser.add_argument('--momentum', action='store_true')
parser.add_argument('--dataset', dest='dataset', type=str, choices=available_datasets, help='dataset to use for the training')
parser.add_argument('--mode', dest='mode', type=str,choices=available_modes, default='train', help='train or test')
parser.add_argument('--epoch', dest='test_epoch', type=int, default=None, help='epoch to be used for testing, mandatory if mode=test')
parser.add_argument('--training-epochs', dest='n_training_epochs', type=int, default=220, help='epoch to be used for training, default 220')
parser.add_argument('--dir', dest='dir', type=str, default=None, help='directory for reading/writing training data and logs')
parser.add_argument('--batch', dest='batch_size', type=int, default=64, help='batch size.')
parser.add_argument('--ngpus', dest='ngpus', type=int, default=1, help='Number of gpus to use.')
parser.add_argument('--sel_gpu', dest='selected_gpu', type=str, default="0", help="one number or two numbers separated by a hyphen")
parser.add_argument('--net', type=str, default='senet50', choices=available_nets, help='Network architecture')
parser.add_argument('--resume', action="store_true", help='resume training')
parser.add_argument('--pretraining', type=str, default=None, help='Pretraining weights, do not set for None, can be vggface or imagenet or a file')
parser.add_argument('--preprocessing', type=str, default='full_normalization', choices=available_normalizations)
parser.add_argument('--augmentation', type=str, default='default', choices=available_augmentations)

# remove
parser.add_argument('--dataset_dir', dest='dataset_dir', type=str, default=None, help='dataset directory to use for the training')

args = parser.parse_args()


import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import sys
import os
import numpy as np
from glob import glob
import re
import tensorflow as tf
import keras
import time
from center_loss import center_loss
from datetime import datetime
from model_build import senet_model_build, vgg16_keras_build, densenet_121_build, mobilenet_224_build, mobilenet_96_build,\
    mobilenet_v3_small_build, mobilenet_v3_large_build, resnet50_build
# vggface_custom_build, mobilenet_64_build, squeezenet_build, shufflenet_224_build, xception_build


sys.path.append("../dataset")
if args.dataset == 'vggface2_age':
    from vgg2_dataset_age import Vgg2DatasetAge as Dataset, NUM_CLASSES
elif args.dataset == 'imdbwiki_age':
    from imdbwiki_dataset_age import IMDBWIKIAge as Dataset, NUM_CLASSES
else:
    print('unknown dataset %s' % args.dataset)
    exit(1)


# Learning Rate
lr = args.lr.split(':')
initial_learning_rate = float(lr[0])  # 0.002
learning_rate_decay_factor = float(lr[1]) if len(lr) > 1 else 0.5
learning_rate_decay_epochs = int(lr[2]) if len(lr) > 2 else 40

# Epochs to train
n_training_epochs = args.n_training_epochs

# Batch size
batch_size = args.batch_size


def step_decay_schedule(initial_lr, decay_factor, step_size):
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch / step_size))

    return keras.callbacks.LearningRateScheduler(schedule, verbose=1)


# Model building
INPUT_SHAPE = None

def get_model():
    global INPUT_SHAPE
    if args.net.startswith("resnet50"):
        if args.net == "resnet50":
            INPUT_SHAPE = (224, 224, 3)
        elif args.net == "resnet50_112":
            INPUT_SHAPE = (112, 112, 3)
        return resnet50_build(INPUT_SHAPE, NUM_CLASSES, args.pretraining)
    elif args.net.startswith('senet') or args.net.startswith('resnet') or args.net.startswith('vgg'):
        INPUT_SHAPE = (112, 112, 3) if args.net.endswith("_112") else (224, 224, 3)
        if args.pretraining.startswith('imagenet'):
            if args.net.startswith('senet') or args.net.startswith('resnet'):
                return senet_model_build(INPUT_SHAPE, NUM_CLASSES, args.pretraining)
            else:
                return vgg16_keras_build(INPUT_SHAPE, NUM_CLASSES, args.pretraining)
        else:
            return vggface_custom_build(INPUT_SHAPE, NUM_CLASSES, args.pretraining, args.net, args.lpf_size)
    elif args.net == 'mobilenet96':
        INPUT_SHAPE = (96, 96, 3)
        return mobilenet_96_build(INPUT_SHAPE, NUM_CLASSES, args.pretraining)
    elif args.net == 'mobilenet224':
        INPUT_SHAPE = (224, 224, 3)
        return mobilenet_224_build(INPUT_SHAPE, NUM_CLASSES, args.pretraining)
    elif args.net == 'mobilenet64_bio':
        INPUT_SHAPE = (64, 64, 3)
        return mobilenet_64_build(INPUT_SHAPE, NUM_CLASSES)
    elif args.net.startswith('mobilenetv3large'):
        if args.net == 'mobilenetv3large':
            INPUT_SHAPE = (224, 224, 3)
        elif args.net == 'mobilenetv3large_112':
            INPUT_SHAPE = (112, 112, 3)
        return mobilenet_v3_large_build(INPUT_SHAPE, NUM_CLASSES, args.pretraining)
    elif args.net == 'mobilenetv3small':
        INPUT_SHAPE = (224, 224, 3)
        return mobilenet_v3_small_build(INPUT_SHAPE, NUM_CLASSES, args.pretraining)
    elif args.net == 'densenet121bc':
        INPUT_SHAPE = (224, 224, 3)
        return densenet_121_build(INPUT_SHAPE, NUM_CLASSES, args.pretraining, args.lpf_size)
    elif args.net.startswith('xception'):
        INPUT_SHAPE = (71, 71, 3) if args.net == 'xception71' else (299, 299, 3)
        return xception_build(INPUT_SHAPE, NUM_CLASSES, args.pretraining, args.lpf_size)
    elif args.net == "shufflenet224":
        INPUT_SHAPE = (224, 224, 3)
        return shufflenet_224_build(INPUT_SHAPE, NUM_CLASSES, args.pretraining)
    elif args.net == "squeezenet":
        INPUT_SHAPE = (224, 224, 3)
        return squeezenet_build(INPUT_SHAPE, NUM_CLASSES, args.pretraining)


# Model creating
gpu_to_use = [str(s) for s in args.selected_gpu.split(',') if s.isdigit()]
if args.ngpus <= 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpu_to_use)
    print("WARNING: Using GPU %s" % os.environ["CUDA_VISIBLE_DEVICES"])
    model, feature_layer = get_model()
else:
    if len(gpu_to_use) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpu_to_use)
    print("WARNING: Using GPU %s" % os.environ["CUDA_VISIBLE_DEVICES"])
    print("WARNING: Using %d gpus" % args.ngpus)
    with tf.device('/cpu:0'):
        model, feature_layer = get_model()
    model = keras.utils.multi_gpu_model(model, args.ngpus)
model.summary()

# model compiling
if args.weight_decay:
    weight_decay = args.weight_decay  # 0.0005
    for layer in model.layers:
        if isinstance(layer, keras.layers.Conv2D) and not isinstance(layer, keras.layers.DepthwiseConv2D) or isinstance(
                layer, keras.layers.Dense):
            layer.add_loss(keras.regularizers.l2(weight_decay)(layer.kernel))
        if hasattr(layer, 'bias_regularizer') and layer.use_bias:
            layer.add_loss(keras.regularizers.l2(weight_decay)(layer.bias))
optimizer = keras.optimizers.sgd(momentum=0.9) if args.momentum else 'sgd'
if args.center_loss:
    loss = center_loss(feature_layer, keras.losses.categorical_crossentropy, 0.9, NUM_CLASSES, 0.01, features_dim=2048)
else:
    loss = keras.losses.categorical_crossentropy if NUM_CLASSES > 1 else keras.losses.mean_squared_error
accuracy_metrics = [keras.metrics.categorical_accuracy] if NUM_CLASSES > 1 else [keras.metrics.mean_squared_error]
model.compile(loss=loss, optimizer=optimizer, metrics=accuracy_metrics)


# Directory creating to store model checkpoints
datetime = datetime.today().strftime('%Y%m%d_%H%M%S')
dirnm = "inference_time_test" if args.mode.endswith('inference') else "trained"
dirnm = os.path.join("..", dirnm)
if not os.path.isdir(dirnm): os.mkdir(dirnm)
argstring = ''.join(sys.argv[1:]).replace('--', '_').replace('=', '').replace(':', '_')
dirnm += '/%s_%s' % (argstring, datetime)
if args.cutout: dirnm += '_cutout'
if args.dir: dirnm = args.dir
if not os.path.isdir(dirnm): os.mkdir(dirnm)
filepath = os.path.join(dirnm, "checkpoint.{epoch:02d}.hdf5")
logdir = dirnm
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
    return max_ep, max_c



# AUGMENTATION 
if args.cutout:
    from cropout_test import CropoutAugmentation
    custom_augmentation = CropoutAugmentation()
elif args.augmentation == 'autoaugment-rafdb':
    from autoaug_test import MyAutoAugmentation
    from autoaugment.rafdb_policies import rafdb_policies
    custom_augmentation = MyAutoAugmentation(rafdb_policies)
elif args.augmentation == 'default':
    from dataset_tools import DefaultAugmentation
    custom_augmentation = DefaultAugmentation()
elif args.augmentation == 'vggface2':
    from dataset_tools import VGGFace2Augmentation
    custom_augmentation = VGGFace2Augmentation()
else:
    custom_augmentation = None


if args.mode.startswith('train'):
    print("TRAINING %s" % dirnm)
    dataset_training = Dataset('train', target_shape=INPUT_SHAPE, augment=True, preprocessing=args.preprocessing, custom_augmentation=custom_augmentation)
    dataset_validation = Dataset('val', target_shape=INPUT_SHAPE, augment=False, preprocessing=args.preprocessing)

    lr_sched = step_decay_schedule(initial_lr=initial_learning_rate,decay_factor=learning_rate_decay_factor, step_size=learning_rate_decay_epochs)
    monitor = 'val_categorical_accuracy' if NUM_CLASSES > 1 else 'val_mean_squared_error'
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, verbose=1, save_best_only=True, monitor=monitor)
    tbCallBack = keras.callbacks.TensorBoard(log_dir=logdir, write_graph=True, write_images=True)
    callbacks_list = [lr_sched, checkpoint, tbCallBack]

    if args.mode == "train_inference":
        batch_size = 1

    if args.resume:
        pattern = filepath.replace('{epoch:02d}', '*')
        epochs = glob(pattern)
        print(pattern)
        print(epochs)
        epochs = [int(x[-8:-5].replace('.', '')) for x in epochs]
        initial_epoch = max(epochs)
        print('Resuming from epoch %d...' % initial_epoch)
        model.load_weights(filepath.format(epoch=initial_epoch))
    else:
        initial_epoch = 0

    model.fit_generator(generator=dataset_training.get_generator(batch_size),
                        validation_data=dataset_validation.get_generator(batch_size),
                        verbose=1, callbacks=callbacks_list, epochs=n_training_epochs, workers=8,
                        initial_epoch=initial_epoch)
    if args.mode == "train_inference":
        print("Warning: TEST ON CPU")
        os.environ["CUDA_VISIBLE_DEVICES"] = ''
        model.fit_generator(generator=dataset_training.get_generator(batch_size),
                            validation_data=dataset_validation.get_generator(batch_size),
                            verbose=1, callbacks=callbacks_list, epochs=n_training_epochs, workers=8,
                            initial_epoch=initial_epoch)
elif args.mode == 'test':
    if args.test_epoch is None:
        args.test_epoch, _ = _find_latest_checkpoint(dirnm)
        print("Using epoch %d" % args.test_epoch)
    model.load_weights(filepath.format(epoch=int(args.test_epoch)))

    # TODO : add inference mode, not in callback

    def evalds(part):
        dataset_test = Dataset(part, target_shape=INPUT_SHAPE, augment=False, preprocessing=args.preprocessing)
        print('Evaluating %s results...' % part)
        result = model.evaluate_generator(dataset_test.get_generator(batch_size), verbose=1, workers=4)
        print('%s results: loss %.3f - accuracy %.3f' % (part, result[0], result[1]))

    evalds('test')
    evalds('val')
    evalds('train')


