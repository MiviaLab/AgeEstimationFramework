import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from vgg2_dataset import PARTITION_TEST, PARTITION_TRAIN, PARTITION_VAL
from cv2 import cv2
from tqdm import tqdm
import os
import pickle
import numpy as np
import csv
from collections import defaultdict

import sys
sys.path.append("../training")
from dataset_tools import DataGenerator, enclosing_square, add_margin

EXT_ROOT = os.path.dirname(os.path.abspath(__file__))

NUM_CLASSES = 1 #101 # 1 # FOR REGRESSION PURPOSE

imdb_ages = None
wiki_ages = None
imdb_identities = None
wiki_identities = None
people_by_identity_imdb = dict()
people_by_identity_wiki = dict()

cleaned_dataset = True
aligned_dataset = True

def check_image_name(image_name):
    return not os.path.splitext(image_name)[0].split("_")[-2].endswith("0-0")

# Loading data from annotations
def _load_ages(metacsv):
    ages = dict()
    identities = dict()
    csv_array = _readcsv(metacsv)
    i = 0
    errors = 0
    for line in csv_array:
        try:
            # change based on csv
            ages[line[2]] = (get_age_label(line[1]), i)
            csv_identity = " ".join(line[7:])
            identities[line[2]] = (csv_identity, i)
            i += 1
        except ValueError:
            errors += 1
    return ages, identities, errors


def _load_imdb_ages(metacsv):
    global imdb_ages
    global imdb_identities
    if imdb_ages is None or imdb_identities is None:
        print("Loading IMDB ages...")
        imdb_ages, imdb_identities, errors = _load_ages(metacsv)
        print("Ages:", len(imdb_ages))
        if errors: print("Errors:", errors)


def _load_wiki_ages(metacsv):
    global wiki_ages
    global wiki_identities
    if wiki_ages is None or wiki_identities is None:
        print("Loading IMDB ages...")
        wiki_ages, wiki_identities, errors = _load_ages(metacsv)
        print("Ages:", len(wiki_ages))
        if errors: print("Errors:", errors)


#  UTILITIES 
def get_age_label(floating_string, precision=3):
    return float(floating_string) if precision is None else np.round(float(floating_string), precision)
    # return int(floating_string)

def _readcsv(csvpath, debug_max_num_samples=None):
    data = list()
    with open(csvpath, newline='', encoding="utf8") as csvfile:
        reader = csv.reader(csvfile, skipinitialspace=True, delimiter=',', quotechar='|')
        i = 0
        for row in reader:
            if debug_max_num_samples is not None and i >= debug_max_num_samples:
                break
            i = i + 1
            data.append(row)
    return np.array(data)


# GETTING 
def get_age_from_wiki(wiki_path_image, metacsv):
    _load_wiki_ages(metacsv)
    try:
        age, i = wiki_ages[wiki_path_image]
        identity, j = wiki_identities[wiki_path_image]
        assert i == j, "Error WIKI loading ages and identities"
        return age, identity
    except KeyError:
        return None, None

def get_age_from_imdb(imdb_path_image, metacsv):
    _load_imdb_ages(metacsv)
    try:
        age, i = imdb_ages[imdb_path_image]
        identity, j = imdb_identities[imdb_path_image]
        assert i == j, "Error IMDB loading ages and identities"
        return age, identity
    except KeyError:
        return None, None

def get_roi(d):
    return [int(d[0]), int(d[1]), int(d[2])-int(d[0]), int(d[3])-int(d[1])]

def get_roi_standard(d):
    return [int(d[0]), int(d[1]), int(d[2]), int(d[3])]


# SPLIT BY IDENTITY 
def get_partition_imdb(identity):
    global people_by_identity_imdb
    return get_partition_identity(identity, people_by_identity_imdb)

def get_partition_wiki(identity):
    global people_by_identity_wiki
    return get_partition_identity(identity, people_by_identity_wiki)

def get_partition_identity(identity_label, people_by_identity):    
    try:
        faces, partition = people_by_identity[identity_label]
        people_by_identity[identity_label] = (faces + 1, partition)
    except KeyError:
        # split 10/10/80 stratified by identity
        l = len(people_by_identity)%10
        if l == 0:
            partition = PARTITION_VAL
        elif l == 1:
            partition = PARTITION_TEST
        else:
            partition = PARTITION_TRAIN
        people_by_identity[identity_label] = (1, partition)
    return partition


# LOAD DATASET 
def _load_imdb(meta, csvmeta, imagesdir):
    return _load_dataset(meta, csvmeta, imagesdir, get_age_from_imdb, get_partition_imdb)

def _load_wiki(meta, csvmeta, imagesdir):
    return _load_dataset(meta, csvmeta, imagesdir, get_age_from_wiki, get_partition_wiki)


def _load_dataset(meta, csvmeta, imagesdir, get_age_function, get_partition_function):
    # discrard: None images, [1,1,1,1] roi, Negative ages, Empty identity string
    data = []
    n_discarded = defaultdict(int)
    for _, item in enumerate(tqdm(meta)):
        image_path = item[2]
        age, identity = get_age_function(image_path, csvmeta)
        complete_image_path = os.path.join(imagesdir, image_path)

        if not check_image_name(os.path.split(image_path)[-1]):
            print("Inconsistent image name:", os.path.split(image_path)[-1])
            n_discarded["Inconsistent name"] += 1
            continue

        # age 0 excluded in order ot avoid BIRTH_DATE and PHOTO_TAKEN overlap
        if age is None or age < 1 or age > 100:
            print("Negative, over 100 or inconsistent age, sample discarderd:", complete_image_path)
            n_discarded["Inconsistent age"] += 1
            continue

        if not len(identity):
            print("Inconsistent identity, sample discaded:", image_path)
            n_discarded["Inconsistent identity"] += 1
            continue

        img = cv2.imread(complete_image_path)

        if img is None:
            print("Unable to read the image:", complete_image_path)
            n_discarded["Unable to read image"] += 1
            continue

        if np.max(img) == np.min(img):
            print('Blank image, sample discarded:', complete_image_path)
            n_discarded["Blank images"] += 1
            continue

        sample_partition = get_partition_function(identity)

        if aligned_dataset:
            roi = get_roi_standard(item[3:7])
        elif cleaned_dataset:
            roi = get_roi(item[3:7])
            roi = enclosing_square(roi)
            roi = add_margin(roi, roi[2]*0.05)
        else:
            roi = [0, 0, img.shape[1], img.shape[0]]
            roi = enclosing_square(roi)
            roi = add_margin(roi, -roi[2]*0.2)

        # age += np.random.choice([1,-1])*np.finfo(np.float32).eps

        example = {
                'img': complete_image_path,
                'label': age,
                'roi': roi, 
                'part': sample_partition
            }

        data.append(example)
    
    total_discarded = sum([v for v in n_discarded.values()])
    print("Data loaded. {} samples out of a total of {} annotations".format(len(data), (len(data) + total_discarded)))
    if n_discarded:
        print("Discarded {} samples:".format(total_discarded))
        for k, v in n_discarded.items():
            print("\t{} : {}".format(k, v))
    return data


def _load_imdbwiki(csvmeta, imagesdir, partition, debug_max_num_samples=None):
    imdb_images_dir = imagesdir.replace('<part>', "imdb")
    wiki_images_dir = imagesdir.replace('<part>', "wiki")
    

    # Load original vggface2 csv containing gender information in 0-1 format
    imdb_csvmeta = csvmeta.replace('<part>', "imdb")
    wiki_csvmeta = csvmeta.replace('<part>', "wiki")
    

    # Load meta from csv
    imdb_meta = _readcsv(imdb_csvmeta, debug_max_num_samples)
    wiki_meta = _readcsv(wiki_csvmeta, debug_max_num_samples)

    # Log read csv
    print("CSV {} read complete: {} samples".format(imdb_csvmeta, len(imdb_meta)))
    print("CSV {} read complete: {} samples".format(wiki_csvmeta, len(wiki_meta)))

    # load datasets
    imdb = _load_imdb(imdb_meta, imdb_csvmeta, imdb_images_dir)
    wiki = _load_wiki(wiki_meta, wiki_csvmeta, wiki_images_dir)

    return imdb, wiki


UNCLEANED_CSVMETA = 'imdbwiki/<part>/<part>_names.csv'
CLEANED_CSVMETA = 'imdbwiki/<part>/<part>_cleaned.csv'
ALIGNED_CSVMETA = 'imdbwiki_aligned/<part>/<part>_aligned.csv'

ALIGNED_IMAGESDIR = 'imdbwiki_aligned/<part>/<part>_crop_aligned'
UNALIGNED_IMAGESDIR = 'imdbwiki/<part>/<part>_crop'

class IMDBWIKIAge:
    def __init__(self,
                 partition='train',
                 imagesdir=ALIGNED_IMAGESDIR,
                 csvmeta=ALIGNED_CSVMETA,
                 target_shape=(224, 224, 3),
                 augment=True,
                 custom_augmentation=None,
                 preprocessing='full_normalization',
                 debug_max_num_samples=None):
        if partition.startswith('train'):
            partition_label = PARTITION_TRAIN
        elif partition.startswith('val'):
            partition_label = PARTITION_VAL
        elif partition.startswith('test'):
            partition_label = PARTITION_TEST
        else:
            raise Exception("unknown partition")

        self.target_shape = target_shape
        self.custom_augmentation = custom_augmentation
        self.augment = augment
        self.gen = None
        self.preprocessing = preprocessing
        print('Loading %s data...' % partition)

        num_samples = "_"+str(debug_max_num_samples) if debug_max_num_samples is not None else ''
        cache_file_name = 'imdbwiki_age_{partition}{num_samples}.cache'.format(partition=partition, num_samples=num_samples)

        if aligned_dataset:
            cache_file_name = "aligned_" + cache_file_name

        if csvmeta == UNCLEANED_CSVMETA:
            print("******IMDB WIKI NOT CLEANED LOADING******")
            cache_file_name = "raw_" + cache_file_name
            global cleaned_dataset
            cleaned_dataset = False

        cache_file_name = os.path.join("dataset_cache", cache_file_name)
        cache_file_name = os.path.join(EXT_ROOT, cache_file_name)
        print("cache file name %s" % cache_file_name)

        try:
            with open(cache_file_name, 'rb') as f:
                self.data = pickle.load(f)
                self.data = self.data[:debug_max_num_samples]
                print("Data loaded. %d samples, from cache" % (len(self.data)))
        except FileNotFoundError:
            print("Loading %s data from scratch" % partition)

            csvmeta = os.path.join(EXT_ROOT, csvmeta)
            imagesdir = os.path.join(EXT_ROOT, imagesdir)

            load_partition = "train" if partition_label == PARTITION_TRAIN or partition_label == PARTITION_VAL else "test"
            imdb, wiki = _load_imdbwiki(csvmeta, imagesdir, load_partition, debug_max_num_samples)
            
            # add merge option "imdb", "wiki", "all"
            self.data = [item for item in imdb if item['part'] == partition_label]
            self.data.extend([item for item in wiki if item['part'] == partition_label])

            print("Entire IMDBWIKI loaded samples: {}".format(len(self.data)))

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


def test_age(partition="test", debug_samples=None):
    print("Partion", partition, debug_samples if debug_samples is not None else '')
    dataset = IMDBWIKIAge(partition=partition,
                            target_shape=(224, 224, 3), 
                            preprocessing='vggface2', 
                            augment=False, 
                            debug_max_num_samples=debug_samples)
    print("Samples in dataset partition", dataset.get_num_samples())

    if people_by_identity_imdb:
        print("IMDB statistics...")
        samples_stats = defaultdict(list)
        for _, identity_data in people_by_identity_imdb.items():
            if identity_data[1] == PARTITION_TRAIN:
                samples_stats["train"].append(identity_data[0])
            elif identity_data[1] == PARTITION_VAL:
                samples_stats["val"].append(identity_data[0])
            elif identity_data[1] == PARTITION_TEST:
                samples_stats["test"].append(identity_data[0])
            else:
                print("Error loading partition", identity_data[0])
        print("Total train {} of different samples {}".format(sum(samples_stats["train"]), len(samples_stats["train"])))
        print("Total val {} of different samples {}".format(sum(samples_stats["val"]), len(samples_stats["val"])))
        print("Total test {} of different samples {}".format(sum(samples_stats["test"]), len(samples_stats["test"])))
        print()
    else:
        print("IMDB loaded from cache...")

    if people_by_identity_wiki:
        print("WIKI statistics...")
        samples_stats = defaultdict(list)
        for _, identity_data  in people_by_identity_wiki.items():
            if identity_data[1] == PARTITION_TRAIN:
                samples_stats["train"].append(identity_data[0])
            elif identity_data[1] == PARTITION_VAL:
                samples_stats["val"].append(identity_data[0])
            elif identity_data[1] == PARTITION_TEST:
                samples_stats["test"].append(identity_data[0])
            else:
                print("Error loading partition", identity_data[0])
        print("Total train {} of different samples {}".format(sum(samples_stats["train"]), len(samples_stats["train"])))
        print("Total val {} of different samples {}".format(sum(samples_stats["val"]), len(samples_stats["val"])))
        print("Total test {} of different samples {}".format(sum(samples_stats["test"]), len(samples_stats["test"])))
        print()
    else:
        print("WIKI loaded from cache...")

    gen = dataset.get_generator(fullinfo=True)

    for batch in tqdm(gen):
        for im, age, path, roi in zip(batch[0], batch[1], batch[2], batch[3]):
            print("Path:", path)
            print("Roi:", roi)
            print("Age:", age, type(age))
            print("Shape:", im.shape)
            facemax = np.max(im)
            facemin = np.min(im)
            im = (255 * ((im - facemin) / (facemax - facemin))).astype(np.uint8)
            cv2.putText(im, "{}".format(np.argmax(age) if NUM_CLASSES > 1 else age), (0, im.shape[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255))
            cv2.imshow('image', im)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                return


if '__main__' == __name__:
    test_age("train", 100)
    # test_age("train")
    # test_age("val")
    # test_age("test")

