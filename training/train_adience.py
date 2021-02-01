

available_nets = ['senet50', 'vgg16', 'densenet121bc', 'xception', 'xception71', 'mobilenet96', 'mobilenet224',
                  'mobilenet64_bio', 'shufflenet224', 'squeezenet', 'mobilenetv3small', 'mobilenetv3large']

available_normalizations = ['z_normalization', 'full_normalization', 'vggface2']
available_augmentations = ['default', 'vggface2', 'no']
available_lpf = [0, 1, 2, 3, 5, 7]

# import argparse
# parser = argparse.ArgumentParser(description='Common training and evaluation.')
# parser.add_argument('--net', type=str, help='Path of the directory which contains weights')
# parser.add_argument('--train_folds', type=str, help='Numbers separated by hyphen')
# parser.add_argument('--val_folds', type=str, help='Numbers separated by hyphen')

# parser.add_argument('--lpf', dest='lpf_size', type=int, choices=available_lpf, default=1, help='size of the lpf filter (1 means no filtering)')
# parser.add_argument('--cutout', action='store_true', help='use cutout augmentation')
# parser.add_argument('--weight_decay', type=float)
# parser.add_argument('--lr', default='0.002', help='Initial learning rate or init:factor:epochs', type=str)
# parser.add_argument('--momentum', action='store_true')
# # parser.add_argument('--epoch', dest='test_epoch', type=int, default=None, help='epoch to be used for testing, mandatory if mode=test')
# parser.add_argument('--training-epochs', dest='n_training_epochs', type=int, default=220, help='epoch to be used for training, default 220')
# parser.add_argument('--batch', dest='batch_size', type=int, default=64, help='batch size.')

# # parser.add_argument('--ngpus', dest='ngpus', type=int, default=1, help='Number of gpus to use.')
# parser.add_argument('--sel_gpu', dest='selected_gpu', type=str, default="0", help="one or more numbers separated by a hyphen")

# parser.add_argument('--pretraining', type=str, default=None, help='Pretraining weights, do not set for None, can be vggface or imagenet or a file')
# parser.add_argument('--preprocessing', type=str, default='full_normalization', choices=available_normalizations)
# parser.add_argument('--augmentation', type=str, default='default', choices=available_augmentations)
# args = parser.parse_args()

import numpy as np
import keras
import tensorflow as tf
import os
from datetime import datetime
from glob import glob
import re
import json

import sys
sys.path.append("../dataset")
from adience_dataset_age import AdienceAge as Dataset, NUM_CLASSES

sys.path.append('keras_vggface/keras_vggface')
from antialiasing import BlurPool

sys.path.append('scratch_models')
from mobile_net_v2_keras import relu6

sys.path.append('../evaluate')
from evaluate_utils import load_keras_model

from model_build import age_relu, Hswish, HSigmoid


custom_objects = {'BlurPool': BlurPool,
                  'relu6': relu6,
                  'age_relu': age_relu,
                  'Hswish': Hswish,
                  'HSigmoid': HSigmoid
                  }


# def build_adience_finetuned(model):
#     # freeze all net, drop last layer, insert categorical layer trainable
#     for layer in model.layers:
#         layer.trainable = False
#     features = model.layers[-2].output

#     # x = keras.layers.Dropout(rate=0.5)(features)
#     # x = keras.layers.Dense(NUM_CLASSES, activation='softmax', name='Logits', trainable=True)(x)

#     x = keras.layers.Dense(NUM_CLASSES, activation='softmax', name='Logits', trainable=True)(features)

#     model = keras.models.Model(model.input, x)
#     return model


# def check_adience_finetuned(model):
#     for layer in model.layers[:-2]:
#         if layer.trainable:
#             model.summary()
#             print("Model unfrozen")
#             return False
#     if not model.layers[-1].trainable:
#         model.summary()
#         print("Top layer not trainable")
#         return False
#     return True
     
# ALL FINETUNING
def build_adience_finetuned(model):
    for layer in model.layers:
        layer.trainable = True
    l1 = model.layers[-2]
    l2 = model.layers[-3]
    l3 = model.layers[-4]
    if l1.output_shape[-1]<20 and l2.output_shape[-1]<20:
        features = l3.output
    else:
        features = l1.output
    if len(features.shape) > 2:
        x = keras.layers.Flatten()(features)
    else:
        x = features
    x = keras.layers.Dropout(rate=0.5)(x)
    x = keras.layers.Dense(NUM_CLASSES, activation='softmax', name='Logits', trainable=True)(x)

    model = keras.models.Model(model.input, x)
    return model

def check_adience_finetuned(model):
    for layer in model.layers:
        if not layer.trainable:
            return False
    return True

def check_param(params, string, typevalue, required=False, default=None, choices=None):
    if required:
        if string not in params:
            raise Exception("{name}parameter is required".format(name=string))
    else:
        if string not in params:
            params[string] = default
            return
    if type(params[string]) is not typevalue:
        raise Exception("Input {name}: {value} not valid".format(name=string, value=params[string]))
    if choices is not None and params[string] not in choices:
        print("Available choices for {name}".format(name=string))
        print(choices)
        raise Exception("Input {name}: {value} not in choices availables".format(name=string, value=params[string]))


def check_train_paramenters(args):
    check_param(args, string="net", typevalue=str, required=True)
    check_param(args, string="out_path", typevalue=str, required=False, default="../fine_tuned_adience")

    check_param(args, string="train_folds", typevalue=list, required=True)
    check_param(args, string="val_folds", typevalue=list, required=True)

    check_param(args, string="datetime", typevalue=str, required=True)

    check_param(args, string="lpf_size", typevalue=int, required=False, default=1, choices=available_lpf)
    check_param(args, string="cutout", typevalue=bool, required=False, default=False)
    check_param(args, string="weight_decay", typevalue=float, required=False, default=0)
    check_param(args, string="lr", typevalue=str, required=False, default="0.002")
    check_param(args, string="momentum", typevalue=bool, required=False, default=False)

    check_param(args, string="resume", typevalue=bool, required=False, default=False)

    check_param(args, string="n_training_epochs", typevalue=int, required=False, default=220)
    check_param(args, string="batch_size", typevalue=int, required=False, default=64)
    check_param(args, string="selected_gpu", typevalue=str, required=False, default="")

    # check_param(args, string="pretraining", typevalue=str, required=True)
    check_param(args, string="preprocessing", typevalue=str, required=False, default="full_normalization", choices=available_normalizations)
    check_param(args, string="augmentation", typevalue=str, required=False, default='default', choices=available_augmentations)
    

def check_net(netdirname):
    net = netdirname.split("_")[1][3:]
    net2 = netdirname.split("_")[3][3:]
    if not (net in available_nets or net2 in available_nets):
        raise Exception("{} net not available!".format(net + " or " + net2))


def lr_string_parse(lr_string):
    lr = lr_string.split(':')
    initial_lr = float(lr[0])  # 0.002
    lr_decay_factor = float(lr[1]) if len(lr) > 1 else 0.5
    lr_decay_epochs = int(lr[2]) if len(lr) > 2 else 40
    return initial_lr, lr_decay_factor, lr_decay_epochs


def step_decay_schedule(initial_lr, decay_factor, step_size):

    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch / step_size))

    return keras.callbacks.LearningRateScheduler(schedule, verbose=1)


def net_path_parse(net_string):
    net_path = os.path.join(net_string, "checkpoint.{epoch:02d}.hdf5")
    pattern = net_path.replace('{epoch:02d}', '*')
    epochs = glob(pattern)
    print(pattern)
    print(epochs)
    epochs = [int(x[-8:-5].replace('.', '')) for x in epochs]
    initial_epoch = max(epochs) if epochs else 0
    return net_path, initial_epoch


def load_model_multiple_gpu(weight_path, gpu_to_use):
    gpu_to_use = [str(s) for s in gpu_to_use.split(',') if s.isdigit()]
    if len(gpu_to_use) <= 1:
        if len(gpu_to_use)>0:
            os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpu_to_use)
        print("WARNING: Using GPU %s" % os.environ["CUDA_VISIBLE_DEVICES"])
        # model = keras.models.load_model(weight_path, custom_objects=custom_objects)
        model, _ = load_keras_model(weight_path)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpu_to_use)
        print("WARNING: Using GPU %s" % os.environ["CUDA_VISIBLE_DEVICES"])
        print("WARNING: Using %d gpus" % len(gpu_to_use))
        with tf.device('/cpu:0'):
            # model = keras.models.load_model(weight_path, custom_objects=custom_objects)
            model, _ = load_keras_model(weight_path)
        model = keras.utils.multi_gpu_model(model, len(gpu_to_use))
    input_shape = tuple(model.input.shape[1:])
    model.summary()
    return model, input_shape


def output_path_generation(args):
    # datetime_str = datetime.today().strftime('%Y%m%d_%H%M%S')
    out_path = args.out_path #os.path.join("..", "fine_tuned_adience")
    out_path = os.path.abspath(out_path)
    if not os.path.isdir(out_path): os.mkdir(out_path)

    # argstring = ''.join(sys.argv[1:]).replace('--', '_').replace('=', '').replace(':', '_')
    # out_path += '/%s_%s' % (argstring, datetime)
    # if args.cutout: out_path += '_cutout'
    # if args.dir: out_path = args.dir
    # if not os.path.isdir(out_path): os.mkdir(out_path)

    pretrained_net = args.net[:-1] if args.net.endswith("/") else args.net
    pretrained_net = os.path.split(pretrained_net)[1]

    check_net(pretrained_net)

    out_path = os.path.join(out_path, args.datetime + "_" + pretrained_net)
    if not os.path.isdir(out_path): os.mkdir(out_path)

    train_fold_string = "_".join(args.train_folds)
    val_fold_string = "_".join(args.val_folds)

    out_path = os.path.join(out_path, "train_folds_{}_val_folds_{}".format(train_fold_string, val_fold_string))
    if not os.path.isdir(out_path): os.mkdir(out_path)

    with open(os.path.join(out_path, "params.json"), "w") as f:
        json.dump(args, f)

    logdir = out_path
    out_path = os.path.join(out_path, "checkpoint.{epoch:02d}.hdf5")
    return out_path, logdir


def deserialize_folds(foldstring):
    fold_format="fold_{fold_number}"
    folds = list()
    for s in foldstring.split(','):
        try:
            f = int(s)
        except ValueError:
            raise Exception("Fold unrecognized: {} not a number".format(s))
        folds.append(fold_format.format(fold_number=f))
    return folds


def run_train(args):
    # check parameterss
    check_train_paramenters(args)
    pretrained_net = args.net[:-1] if args.net.endswith("/") else args.net
    pretrained_net = os.path.split(pretrained_net)[1]
    check_net(pretrained_net)

    # Learning Rate
    initial_learning_rate, learning_rate_decay_factor, learning_rate_decay_epochs = lr_string_parse(args.lr)

    # Epochs to train
    n_training_epochs = args.n_training_epochs

    # Batch size
    batch_size = args.batch_size

    # model tracing
    net_path, initial_epoch = net_path_parse(args.net)
    if net_path.split("/")[-3].startswith("fine_tuned_adience") and args.resume:
        print('Resuming finetuning {} from epoch {}...'.format("fine_tuned_adience".upper(), initial_epoch))
    else:
        print('Starting finetuning from epoch {}...'.format(initial_epoch))

    print("Best checkpoint path: ", net_path.format(epoch=initial_epoch))

    # use of GPU and model loading
    model, input_shape = load_model_multiple_gpu(net_path.format(epoch=initial_epoch), args.selected_gpu)

    # modify model to perform finetuning
    model = build_adience_finetuned(model)
    assert check_adience_finetuned(model), "Error in building network for finetuning"

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

    loss = keras.losses.categorical_crossentropy
    accuracy_metrics = [keras.metrics.categorical_accuracy]

    model.compile(loss=loss, optimizer=optimizer, metrics=accuracy_metrics)

    out_path, logdir = output_path_generation(args)

    model.summary()

    # Augmentation loading
    if args.cutout:
        from cropout_test import CropoutAugmentation
        custom_augmentation = CropoutAugmentation()
    elif args.augmentation == 'default':
        from dataset_tools import DefaultAugmentation
        custom_augmentation = DefaultAugmentation()
    elif args.augmentation == 'vggface2':
        from dataset_tools import VGGFace2Augmentation
        custom_augmentation = VGGFace2Augmentation()
    else:
        custom_augmentation = None

    # load dataset
    train_folds = args.train_folds if type(args.train_folds) is list else deserialize_folds(args.train_folds)
    val_folds = args.val_folds if type(args.val_folds) is list else deserialize_folds(args.val_folds)
    print("Loading datasets...")
    print("Train folds:", train_folds)
    print("Validation folds:", val_folds)
    print("Input shape:", input_shape, type(input_shape))
    print("Preprocessing:", args.preprocessing)
    print("Augmentation:", custom_augmentation)
    dataset_training = Dataset(train_folds,
                                target_shape=input_shape,
                                augment=False,
                                preprocessing=args.preprocessing,
                                custom_augmentation=custom_augmentation)
    if val_folds is not None:
        dataset_validation = Dataset(val_folds,
                                target_shape=input_shape,
                                augment=False,
                                preprocessing=args.preprocessing)  
    else:
        dataset_validation = None
    # select train initial epoch
    train_initial_epoch = initial_epoch if args.resume else 0

    # Training
    print("Training out path", out_path)
    print("Training parameters:")
    for p, v in args.items():
        print(p, ":", v)
    lr_sched = step_decay_schedule(initial_lr=initial_learning_rate,decay_factor=learning_rate_decay_factor, step_size=learning_rate_decay_epochs)
    monitor = 'val_categorical_accuracy'
    checkpoint = keras.callbacks.ModelCheckpoint(out_path, verbose=1, save_best_only=True, monitor=monitor)
    tbCallBack = keras.callbacks.TensorBoard(log_dir=logdir, write_graph=True, write_images=True)
    # early_stopping = keras.callbacks.EarlyStopping(monitor=monitor, patience=10, min_delta=1, verbose=1)
    callbacks_list = [lr_sched, checkpoint, tbCallBack] #, early_stopping]
    # TODO epochs = 10
    # TODO ? lr = 0.001
    # TODO graph tensorboard store

    model.fit_generator(generator=dataset_training.get_generator(batch_size),
                        validation_data=dataset_validation.get_generator(batch_size),
                        verbose=1,
                        callbacks=callbacks_list,
                        epochs=n_training_epochs,
                        workers=8,
                        initial_epoch=train_initial_epoch)

    if len(val_folds)==0:
        model.save(out_path.format(epoch=n_training_epochs))


# def _find_latest_checkpoint(d):
#     ep_re = re.compile('checkpoint.([0-9]+).hdf5')
#     all_checks = glob(os.path.join(d, '*'))
#     max_ep = 0
#     max_c = None
#     for c in all_checks:
#         epoch_num = re.search(ep_re, c)
#         if epoch_num is not None:
#             epoch_num = int(epoch_num.groups(1)[0])
#             if epoch_num > max_ep:
#                 max_ep = epoch_num
#                 max_c = c
#     return max_ep, max_c


