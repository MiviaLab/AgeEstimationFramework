import sys
from collections import defaultdict
import csv
import numpy as np
import pickle
import os
from tqdm import tqdm
from cv2 import cv2
from scipy.io import loadmat

from vgg2_dataset import PARTITION_TEST, PARTITION_TRAIN, PARTITION_VAL
sys.path.append("../training")
from dataset_tools import DataGenerator, enclosing_square, add_margin


# import warnings
# warnings.filterwarnings('ignore', category=FutureWarning)

NUM_CLASSES = 1  # FOR REGRESSION PURPOSE
CROPPED_SUFFIX = "_face.jpg" # FOR NOT ALIGNED DATASET
EXT_ROOT = os.path.dirname(os.path.abspath(__file__))
lap_ages = defaultdict(dict)

def get_roi_lap(d):
    return (int(d[0]), int(d[1]), int(d[2]), int(d[3]))

# Loading structured data
def _get_structured_lap_meta(metacsv):
    data = dict()
    csv_array = _readcsv(metacsv)
    for line in csv_array[1:]: # header exclusion
        data[line[0]] = {
            "cropped" : line[0],
            "apparent_age": line[2],
            "apparent_std": line[3],
            "real_age": line[4],
            "roi": get_roi_lap(line[5:9]),
        }
    return data


def structured_lap_data_wrapper(partition):
    metacsv = 'chalearn_aligned/gt_avg_<part>.csv'
    metapart = get_metapartition_label(get_partition_label(partition))
    metacsv = os.path.join(EXT_ROOT, metacsv.replace("<part>", metapart))
    return _get_structured_lap_meta(metacsv)


def _load_ages(metacsv, partition):
    global lap_ages
    if lap_ages is None or partition not in lap_ages or lap_ages[partition] is None:
        lap_ages[partition] = _get_structured_lap_meta(metacsv)


def get_age_label(floating_string, precision=3):
    return float(floating_string) if precision is None else np.round(float(floating_string), precision)


def _readcsv(csvpath, debug_max_num_samples=None):
    data = list()
    with open(csvpath, newline='', encoding="utf8") as csvfile:
        reader = csv.reader(csvfile, skipinitialspace=True,
                            delimiter=',', quotechar='|')
        i = 0
        for row in reader:
            if debug_max_num_samples is not None and i >= debug_max_num_samples:
                break
            i = i + 1
            data.append(row)
    return np.array(data)


def get_partition_label(partition):
    if partition == 'train':
        return PARTITION_TRAIN
    elif partition == 'val':
        return PARTITION_VAL
    elif partition == 'test':
        return PARTITION_TEST
    else:
        raise Exception("unknown partition")

# metapartition alias name of directory of dataset
def get_metapartition_label(partition_label):
    if partition_label == PARTITION_TRAIN:
        return 'train'
    elif partition_label == PARTITION_VAL:
        return 'valid'
    elif partition_label == PARTITION_TEST:
        return 'test'
    else:
        raise Exception("unknown meta partition")


def get_age_from_lap(path, metacsv, partition):
    _load_ages(metacsv, partition)
    try:
        cropped = lap_ages[partition][path]['cropped']
        age = lap_ages[partition][path]['apparent_age']
        roi = lap_ages[partition][path]['roi']
        return cropped, age, roi
    except KeyError:
        return None, None, None


def get_prebuilt_roi(filepath):
    if filepath.endswith(CROPPED_SUFFIX):
        filepath = filepath[:-len(CROPPED_SUFFIX)]
    filepath = filepath + ".mat"
    startx, starty, endx, endy, _, _ = loadmat(filepath)['fileinfo']['face_location'][0][0][0]
    roi = (startx, starty, endx-startx, endy-starty)
    roi = enclosing_square(roi)
    roi = add_margin(roi, 0.2)
    return roi



# Load dataset
def _load_dataset(meta, csvmeta, imagesdir, partition, cropped=True):
    data = []
    n_discarded = 0
    for item in tqdm(meta[1:]): # header exclusion
        image_path = item[0]
        cropped_image_path, apparent_age, roi = get_age_from_lap(image_path, csvmeta, partition)
        complete_image_path = os.path.join(imagesdir, cropped_image_path if cropped else image_path)
        partition_label = get_partition_label(partition)

        img = cv2.imread(complete_image_path)

        if img is None:
            print("Unable to read the image:", complete_image_path)
            n_discarded += 1
            continue

        if np.max(img) == np.min(img):
            print('Blank image, sample discarded:', complete_image_path)
            n_discarded += 1
            continue

        if roi == (0, 0, 0, 0) or roi is None:
            # print("No face detected, sample discarded: ", complete_image_path)
            # n_discarded += 1
            # continue
            print("No face detected, entire sample added: ", complete_image_path)
            roi = (0, 0, img.shape[1], img.shape[0])

        example = {
            'img': complete_image_path,
            'label': get_age_label(apparent_age),
            'roi': roi,
            'part': partition_label
        }

        data.append(example)
    print("Data loaded. {} samples ({} discarded)".format(len(data), n_discarded))
    return data


def _load_lap(csvmeta, imagesdir, partition, debug_max_num_samples=None):
    metapartition = get_metapartition_label(get_partition_label(partition))
    print("Directory partition:", metapartition)
    lap_partition_dir = imagesdir.replace('<part>', metapartition)
    lap_partition_csv = csvmeta.replace('<part>', metapartition)
    lap_partition_meta = _readcsv(lap_partition_csv, debug_max_num_samples)
    print("CSV {} read complete: {} samples".format(lap_partition_csv, len(lap_partition_meta)))
    return _load_dataset(lap_partition_meta, lap_partition_csv, lap_partition_dir, partition)

# Original, no roi
# imagesdir='chalearn/<part>',
# csvmeta='chalearn/gt_avg_<part>.csv',

# Aligned, roi
# imagesdir='chalearn_aligned/<part>',
# csvmeta='chalearn_aligned/gt_avg_<part>.csv',

class LAPAge:
    def __init__(self,
                 partition='train',
                 imagesdir='chalearn_aligned/<part>',
                 csvmeta='chalearn_aligned/gt_avg_<part>.csv',
                 target_shape=(224, 224, 3),
                 augment=True,
                 custom_augmentation=None,
                 preprocessing='full_normalization',
                 method='apparent',
                 debug_max_num_samples=None):
        
        

        self.target_shape = target_shape
        self.custom_augmentation = custom_augmentation
        self.augment = augment
        self.gen = None
        self.preprocessing = preprocessing
        print('Loading %s data...' % partition)

        num_samples = "_" + str(debug_max_num_samples) if debug_max_num_samples is not None else ''
        cache_file_name = 'lap_age_{method}_{partition}{num_samples}.cache'.format(method=method,partition=partition, num_samples=num_samples)
        cache_file_name = os.path.join("dataset_cache", cache_file_name)
        cache_file_name = os.path.join(EXT_ROOT, cache_file_name)
        print("cache file name %s" % cache_file_name)

        try:
            with open(cache_file_name, 'rb') as f:
                self.data = pickle.load(f)
                self.data = self.data[:debug_max_num_samples]
                print("Data loaded. %d samples, from cache" % (len(self.data)))
        except FileNotFoundError:

            if partition == "all":
                self.data = list()
                for partition in ["train", "val", "test"]:
                    print("Loading %s data from scratch" % partition)
                    csvmeta = os.path.join(EXT_ROOT, csvmeta)
                    imagesdir = os.path.join(EXT_ROOT, imagesdir)
                    self.data.extend(_load_lap(csvmeta, imagesdir, partition, debug_max_num_samples))
            else:
                print("Loading %s data from scratch" % partition)
                csvmeta = os.path.join(EXT_ROOT, csvmeta)
                imagesdir = os.path.join(EXT_ROOT, imagesdir)
                self.data = _load_lap(csvmeta, imagesdir, partition, debug_max_num_samples)

            # serialization in pickle file
            with open(cache_file_name, 'wb') as f:
                print("Pickle dumping")
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


def test_age(partition="all", debug_samples=None):
    print("Partion", partition, debug_samples if debug_samples is not None else '')
    dataset = LAPAge(partition=partition,
                        target_shape=(224, 224, 3),
                        preprocessing='vggface2',
                        augment=False,
                        debug_max_num_samples=debug_samples)
    print("Samples in dataset partition", dataset.get_num_samples())

    if lap_ages:
        print("LAP CHALEARN statistics...")
        if "train" in lap_ages:
            print("Total train samples {}".format(len(lap_ages["train"])))
        else:
            print("Total train samples 0")
        if "val" in lap_ages:
            print("Total val samples {}".format(len(lap_ages["val"])))
        else:
            print("Total val samples 0")
        if "test" in lap_ages:
            print("Total test {}".format(len(lap_ages["test"])))
        else:
            print("Total test samples 0")
        print()
    else:
        print("LAP DATASET loaded from cache...")

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


if '__main__' == __name__:
    test_age("train")
    test_age("val")
    test_age("test")
