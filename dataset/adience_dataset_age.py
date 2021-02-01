from collections import defaultdict
from tqdm import tqdm
from cv2 import cv2
import os
import pickle
import numpy as np
import os
import sys
import csv
from ast import literal_eval

from adience_utilities import get_structured_adience_meta, get_metafold, parse_landmarks, get_roi_from_landmarks
from adience_utilities import IMAGE_PATH, ANNOTATION_PATH, LANDMARKS_PATH, AGE_CLASSES as CLASSES

sys.path.append("../training")
from dataset_tools import DataGenerator, enclosing_square, add_margin

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

NUM_CLASSES = len(CLASSES)  # 8
EXT_ROOT = os.path.dirname(os.path.abspath(__file__))

adience_data = defaultdict(dict)

SHIFT=0.15

def _load_adience(imagespath, metatxt, foldnumber, landmarkspath):
    global adience_data
    if adience_data is None or foldnumber not in adience_data:
        adience_data[foldnumber] = get_structured_adience_meta(
            imagespath, metatxt, foldnumber, landmarkspath)


def _load_dataset(imagespath, metatxt, foldnumber, landmarkspath):
    # TODO debug samples add
    _load_adience(imagespath, metatxt, foldnumber, landmarkspath)

    data = []
    n_discarded = 0
    n_label_discarded = 0

    for image_path, image_data in tqdm(adience_data[foldnumber].items()):

        img = cv2.imread(image_path)

        if img is None:
            print("Unable to read the image:", image_path)
            n_discarded += 1
            continue

        if np.max(img) == np.min(img):
            print('Blank image, sample discarded:', image_path)
            n_discarded += 1
            continue
        

        if type(image_data['age_label']) is not int:
            n_discarded += 1  # integers and None
            continue

        # face detection from landmarks 
        landmarks = parse_landmarks(image_data['landmarks'])
        roi = get_roi_from_landmarks(landmarks, img.shape[1], img.shape[0])

        # add padding
        roi = enclosing_square(roi)
        roi = add_margin(roi, 0.1*roi[2])
        roi = list(roi)
        roi[1] -= SHIFT*roi[2]
        roi = tuple(roi)
        example = {
            'img': image_path,
            'label': image_data['age_label'],
            'roi': roi,
            'fold': foldnumber
        }

        data.append(example)
    print("Data loaded. {} samples, {} discarded for wrong image, {} discarded for wrong label type".format(
        len(data), n_discarded, n_label_discarded))
    return data


class AdienceAge:
    def __init__(self,
                 fold='fold_0',
                 imagesdir="adience_dataset",
                 target_shape=(224, 224, 3),
                 augment=False,
                 custom_augmentation=None,
                 preprocessing='no_normalization',
                 debug_max_num_samples=None):

        self.target_shape = target_shape
        self.custom_augmentation = custom_augmentation
        self.augment = augment
        self.gen = None
        self.preprocessing = preprocessing

        try:
            parsed_fold = type(literal_eval(fold))
        except (SyntaxError, ValueError):
            parsed_fold = None
                
        if type(fold) is list or type(fold) is tuple:
            foldnumbers = [get_metafold(tmpfold) for tmpfold in fold]
        elif parsed_fold is not None and (type(parsed_fold) is list or type(parsed_fold) is tuple):
            foldnumbers = [get_metafold(tmpfold) for tmpfold in parsed_fold]
        else:
            foldnumbers = [get_metafold(fold)]
        print('Loading data from folds: {}'.format(foldnumbers))

        foldstrings = ["fold_{}".format(f) for f in foldnumbers]
        cache_file_name = 'dataset_cache/adience_age_shift{shift}_{fold}.cache'.format(fold="_".join(foldstrings), shift=SHIFT)
        cache_file_name = os.path.join(EXT_ROOT, cache_file_name)
        print("cache file name {}".format(cache_file_name))

        try:
            with open(cache_file_name, 'rb') as f:
                self.data = pickle.load(f)[:debug_max_num_samples]
                print("Data loaded. {} samples, from cache".format(len(self.data)))
        except FileNotFoundError:
            print("Loading {} data from scratch".format(fold))

            self.data = list()
            for foldnumber in foldnumbers:
                imagespath = IMAGE_PATH.replace("<images_dir>", imagesdir)
                metatxt = ANNOTATION_PATH.replace("<images_dir>", imagesdir).replace("<fold_number>", str(foldnumber))
                landmarkspath = LANDMARKS_PATH.replace("<images_dir>", imagesdir)

                imagespath = os.path.join(EXT_ROOT, imagespath)
                metatxt = os.path.join(EXT_ROOT, metatxt)
                landmarkspath = os.path.join(EXT_ROOT, landmarkspath)

                # add support to debug max samples on load
                self.data += _load_dataset(imagespath, metatxt, foldnumber, landmarkspath)

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



def test_age(fold, debug_samples=None):
    print("Partition", fold, debug_samples if debug_samples is not None else '')
    dataset = AdienceAge(fold=fold,
                         target_shape=(224, 224, 3),
                         preprocessing='vggface2',
                         augment=False,
                         debug_max_num_samples=debug_samples)
    print("Samples in dataset fold", dataset.get_num_samples())

    gen = dataset.get_generator(fullinfo=True)

    for batch in tqdm(gen):
        for im, age_one_hot, path, roi in zip(batch[0], batch[1], batch[2], batch[3]):
            print(age_one_hot)
            age = CLASSES[np.argmax(age_one_hot)]
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
    test_age("fold_0", 100)
    # test_age(["fold_1", "fold_2", "fold_3"])
    # test_age("fold_1")
    # test_age("fold_2")
    # test_age("fold_3")
    # test_age("fold_4")
