import csv
from tqdm import tqdm
import os
from cv2 import cv2
import numpy as np
import pickle

import sys
sys.path.append("../training")
from dataset_tools import enclosing_square, add_margin, DataGenerator

PARTITION_TEST = 2
NUM_CLASSES = 1
EXT_ROOT = os.path.dirname(os.path.abspath(__file__))


def get_age_label(floating_string, precision=3):
    return float(floating_string) if precision is None else np.round(float(floating_string), precision)


def _readcsv(csvpath, debug_max_num_samples=None):
    data = []
    with open(csvpath, newline='', encoding="utf8") as csvfile:
        reader = csv.reader(csvfile, skipinitialspace=True, delimiter=',', quotechar='|')
        i = 0
        for row in reader:
            if debug_max_num_samples is not None and i >= debug_max_num_samples:
                break
            i = i + 1
            data.append(row)
    return np.array(data)

# csvmeta = gender-access/lfw_cropped.csv
# csvmeta = gender-access/lfw_theirs.csv
def _load_lfw_age(csvmeta, imagesdir, debug_max_num_samples=None):
    meta = _readcsv(csvmeta, debug_max_num_samples)
    print('csv %s read complete: %d.' % (csvmeta, len(meta)))
    data = []
    n_discarded = 0
    for d in tqdm(meta):
        path = os.path.join(imagesdir, d[2])
        age = get_age_label(d[1])
        img = cv2.imread(path)
        if img is not None:
            roi = [16, 16, img.shape[0]-32, img.shape[1]-32]
            example = {
                'img': path,
                'label': age,
                'roi': roi,
                'part': PARTITION_TEST
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


class LFWPlusDatasetAge:
    def __init__(self, partition='test',
                imagesdir='gender-access/lfw_cropped',
                csvmeta='gender-access/lfw_theirs.csv',
                target_shape=(256, 256, 3),
                augment=False,
                custom_augmentation=None,
                preprocessing='full_normalization',
                debug_max_num_samples=None,
                cache_dir=None):
                
        if not partition.startswith('test'):
            raise Exception("unknown partition")

        self.target_shape = target_shape
        self.custom_augmentation = custom_augmentation
        self.augment = augment
        self.gen = None
        self.preprocessing = preprocessing
        print('Loading %s data...' % partition)

        num_samples = "_" + str(debug_max_num_samples) if debug_max_num_samples is not None else ''
        cache_file_name = 'lfwplus_age_{partition}{num_samples}.cache'.format(partition=partition, num_samples=num_samples)
        
        if cache_dir is not None:
            if not os.path.isdir(cache_dir):
                os.makedirs(cache_dir, exist_ok=True)
            cache_file_name = os.path.join(cache_dir, cache_file_name)
        else:
            cache_file_name = os.path.join("dataset_cache", cache_file_name)
            cache_file_name = os.path.join(EXT_ROOT, cache_file_name)

        print(cache_dir)
        print("cache file name %s" % cache_file_name)
        try:
            with open(cache_file_name, 'rb') as f:
                self.data = pickle.load(f)
                self.data = self.data[:debug_max_num_samples]
                print("Data loaded. %d samples, from cache" % (len(self.data)))
        except FileNotFoundError:
            print("Loading %s data from scratch..." % partition)
            csvmeta = os.path.join(EXT_ROOT, csvmeta)
            imagesdir = os.path.join(EXT_ROOT, imagesdir)
            self.data = _load_lfw_age(csvmeta, imagesdir, debug_max_num_samples)
            with open(cache_file_name, 'wb') as f:
                print("Pickle dumping...")
                pickle.dump(self.data, f)

    def get_generator(self, batch_size=64, fullinfo=False):
        if self.gen is None:
            self.gen = DataGenerator(self.data, self.target_shape, with_augmentation=self.augment,
                                     custom_augmentation=self.custom_augmentation, batch_size=batch_size,
                                     num_classes=self.get_num_classes(), preprocessing=self.preprocessing,
                                     fullinfo=fullinfo)
        return self.gen

    def get_num_classes(self):
        return NUM_CLASSES

    def get_num_samples(self):
        return len(self.data)


def test_age(partition="test", debug_samples=None):
    print("Partion", partition, debug_samples if debug_samples is not None else '')
    dataset = LFWPlusDatasetAge(partition=partition,
                                target_shape=(224, 224, 3),
                                preprocessing='vggface2',
                                augment=False,
                                debug_max_num_samples=debug_samples)
    print("Samples in dataset partition", dataset.get_num_samples())

    gen = dataset.get_generator(fullinfo=True)

    for batch in tqdm(gen):
        for im, age, path, roi in zip(batch[0], batch[1], batch[2], batch[3]):
            print("Path:", path)
            print("Roi:", roi)
            print("Shape:", im.shape)
            facemax = np.max(im)
            facemin = np.min(im)
            im = (255 * ((im - facemin) / (facemax - facemin))).astype(np.uint8)
            cv2.putText(im, "{}".format(
                age), (0, im.shape[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255))
            cv2.imshow("image", im)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                return

if __name__ == "__main__":
    test_age()