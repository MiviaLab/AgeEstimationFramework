import numpy as np
import time
import random
import cv2
import sys
import os
import keras
from tqdm import tqdm

sys.path.append("../training")

from dataset_tools import enclosing_square, add_margin, cut
from dataset_tools import _readcsv
from dataset_tools import DataGenerator, VGGFace2Augmentation

from six.moves import cPickle as pickle

NUM_CLASSES = 8631 + 500

vgg2ids = None
ids2vgg = None


def _load_identities(idmetacsv):
    global vgg2ids
    global ids2vgg
    if ids2vgg is None:
        vgg2ids = {}
        ids2vgg = []
        arr = _readcsv(idmetacsv)
        i = 0
        for line in arr:
            try:
                vggnum = int(line[0][1:])
                vgg2ids[vggnum] = (line[1], i)
                ids2vgg.append((line[1], vggnum))
                i += 1
            except Exception as e:
                pass
        print(len(ids2vgg), len(vgg2ids), NUM_CLASSES)
        assert (len(ids2vgg) == NUM_CLASSES)
        assert (len(vgg2ids) == NUM_CLASSES)


def get_id_from_vgg2(vggidn, idmetacsv='vggface2/identity_meta.csv'):
    _load_identities(idmetacsv)
    try:
        return vgg2ids[vggidn]
    except KeyError:
        print('ERROR: n%d unknown' % vggidn)
        return 'unknown', -1


def get_vgg2_identity(idn, idmetacsv='vggface2/identity_meta.csv'):
    _load_identities(idmetacsv)
    try:
        return ids2vgg[idn]
    except IndexError:
        print('ERROR: %d unknown', idn)
        return 'unknown', -1


PARTITION_TRAIN = 0
PARTITION_VAL = 1
PARTITION_TEST = 2

by_identity = {}


def get_partition(category_label):
    global by_identity
    try:
        by_identity[category_label] += 1
    except KeyError:
        by_identity[category_label] = 1
    l = by_identity[category_label]
    # split 10/10/80 stratified by identity
    l = (l - 1) % 10
    if l == 0:
        partition = PARTITION_TEST
    elif l == 1:
        partition = PARTITION_VAL
    else:
        partition = PARTITION_TRAIN
    return partition


def _load_vgg2(csvmeta, imagesdir, partition, defer_image_loading=True):
    imagesdir = imagesdir.replace('<part>', partition)
    csvmeta = csvmeta.replace('<part>', partition)
    meta = _readcsv(csvmeta)
    print('csv %s read complete: %d.' % (csvmeta, len(meta)))
    idmetacsv = os.path.join(os.path.dirname(csvmeta), 'identity_meta.csv')
    data = []
    n_discarded = 0
    for n, d in enumerate(tqdm(meta)):
        idname, category_label = get_id_from_vgg2(int(d[3]), idmetacsv)
        path = os.path.join(imagesdir, '%s' % (d[2]))
        img = cv2.imread(path)
        roi = [int(x) for x in d[4:8]]
        roi = enclosing_square(roi)
        roi = add_margin(roi, 0.2)
        partition = get_partition(category_label)

        if img is not None:
            example = {
                'img': path,
                'label': category_label,
                'roi': roi,
                'part': partition
            }
            if np.max(img) == np.min(img):
                print('Warning, blank image: %s!' % path)
            else:
                data.append(example)
        else:  # img is None
            print("WARNING! Unable to read %s" % path)
            n_discarded += 1
    print("Data loaded. %d samples (%d discarded)" % (len(data), n_discarded))
    return data


class Vgg2Dataset:
    def __init__(self, partition='train', imagesdir='vggface2_data/<part>', csvmeta='vggface2/<part>.detected.csv',
                 target_shape=(224, 224, 3), augment=True, custom_augmentation=None, preprocessing='full_normalization',
                 debug_max_num_samples=None):
        if partition.startswith('train'):
            partition = PARTITION_TRAIN
        elif partition.startswith('val'):
            partition = PARTITION_VAL
        elif partition.startswith('test'):
            partition = PARTITION_TEST
        else:
            raise Exception("unknown partition")

        self.target_shape = target_shape
        self.custom_augmentation = custom_augmentation
        self.augment = augment
        self.gen = None
        self.preprocessing = preprocessing
        print('Loading data...')

        cache_file_name = '%s%s.cache' % (csvmeta.replace('/', '_').replace('<part>', 'part'),
                                          '.' + str(debug_max_num_samples) if debug_max_num_samples is not None else '')
        try:
            with open(cache_file_name, 'rb') as f:
                self.data = pickle.load(f)
                self.data = self.data[:debug_max_num_samples]
                self.data = [x for x in self.data if x['part'] == partition]
                print("Data loaded. %d samples, from cache" % (len(self.data)))
        except FileNotFoundError:
            self.data = _load_vgg2(csvmeta, imagesdir, 'train')
            # self.data += _load_vgg2(csvmeta, imagesdir, 'test')
            with open(cache_file_name, 'wb') as f:
                pickle.dump(self.data, f)

    def get_num_samples(self):
        return self.data.shape[0]

    def get_num_classes(self):
        return NUM_CLASSES

    def get_generator(self, batch_size=64):
        if self.gen is None:
            self.gen = DataGenerator(self.data, self.target_shape, with_augmentation=self.augment,
                                     custom_augmentation=self.custom_augmentation, batch_size=batch_size,
                                     num_classes=self.get_num_classes(), preprocessing=self.preprocessing)
        return self.gen


def test1():
    print('Training')
    dt = Vgg2Dataset(target_shape=(224, 224, 3), preprocessing='vggface2',
                     custom_augmentation=VGGFace2Augmentation(), debug_max_num_samples=None)
    # print('Test')
    # dv = Vgg2Dataset('test',target_shape=(224,224,3), preprocessing='full_normalization', debug_max_num_samples=None, augment=False)

    print('Now generating from training set')
    gen = dt.get_generator()
    i = 0
    while True:
        print(i)
        i += 1
        for batch in tqdm(gen):
            for im, identity in zip(batch[0], batch[1]):
                identity = np.argmax(identity)
                facemax = np.max(im)
                facemin = np.min(im)
                im = (255 * ((im - facemin) / (facemax - facemin))).astype(np.uint8)
                cv2.putText(im, "%d %s" % (identity, get_vgg2_identity(identity)[0]), (0, im.shape[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255))
                cv2.imshow('vggface2 image', im)
                cv2.waitKey(0)


if '__main__' == __name__:
    test1()
