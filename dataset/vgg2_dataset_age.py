from cv2 import cv2
from tqdm import tqdm
import os
import pickle
import numpy as np
import csv
import sys
from collections import defaultdict

from vgg2_dataset import Vgg2Dataset, VGGFace2Augmentation, get_id_from_vgg2
from vgg2_dataset import PARTITION_TEST, PARTITION_TRAIN, PARTITION_VAL

sys.path.append("../training")
from dataset_tools import enclosing_square, add_margin
from dataset_tools import DataGenerator

EXT_ROOT = os.path.dirname(os.path.abspath(__file__))

NUM_CLASSES = 1 # FOR REGRESSION PURPOSE

vgg2age = None
people_by_identity = dict()

# Loading
# people_by_age = defaultdict(lambda : defaultdict(tuple))
# def _load_identities(idmetacsv):
#     global vgg2gender
#     if vgg2gender is None:
#         vgg2gender = dict()
#         csv_array = _readcsv(idmetacsv)
#         i = 0
#         errors = 0
#         for line in csv_array:
#             try:
#                 vgg2gender[int(line[0][1:])] = (get_gender_label(line[-1]), i)
#                 i += 1
#             except ValueError:
#                 print("Error load line", line)
#                 errors += 1
#         print("Identities:", len(vgg2gender))
#         if errors:
#             print("Errors:", errors)


def _load_ages(idmetacsv):
    global vgg2age
    if vgg2age is None:
        vgg2age = dict()
        csv_array = _readcsv(idmetacsv)
        i = 0
        errors = 0
        for line in csv_array:
            try:
                vgg2age[line[0]] = (get_age_label(line[-1]), i)
                i += 1
            except ValueError:
                errors += 1
        print("Ages:", len(vgg2age))
        if errors:
            print("Errors:", errors)



# Getting
def get_age_label(floating_string, precision=3):
    return float(floating_string) if precision is None else np.round(float(floating_string), precision)


def get_age_fromvgg2(vgg_path, metacsv):
    _load_ages(metacsv)
    try:
        return vgg2age[vgg_path]
    except KeyError:
        # print('ERROR: {} unknown'.format(vgg_path))
        return None, -1

def get_roi(d):
    roi = [int(x) for x in d]
    roi = enclosing_square(roi)
    return add_margin(roi, 0.2)


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


def _load_vgg2(vgg2_csvmeta, imagesdir, partition, age_csvmeta, debug_max_num_samples=None):
    imagesdir = imagesdir.replace('<part>', partition)

    # Load original vggface2 csv containing gender information in 0-1 format
    vgg2_csvmeta = vgg2_csvmeta.replace('<part>', partition)
    vgg2_meta = _readcsv(vgg2_csvmeta, debug_max_num_samples)
    print('csv %s read complete: %d.' % (vgg2_csvmeta, len(vgg2_meta)))
    idmetacsv = os.path.join(os.path.dirname(vgg2_csvmeta), 'identity_meta.csv')
    
    # Load vggface2 age csv
    age_csvmeta = age_csvmeta.replace('<part>', partition)
    # age_meta = _readcsv(age_csvmeta, debug_max_num_samples)

    data = []
    n_discarded = 0
    for _, d in enumerate(tqdm(vgg2_meta)):
        _, vgg_identity = get_id_from_vgg2(int(d[3]), idmetacsv)
        age_label = get_age_fromvgg2(d[2], age_csvmeta)[0]

        if age_label is None:
            continue
        
        path = os.path.join(imagesdir, '%s' % (d[2]))
        img = cv2.imread(path)

        if img is not None:
            # if age_meta is not None:
            #     age_label = get_age_fromvgg2(d[2], age_csvmeta)[0]
            #     if age_label is None:
            #         continue
            #     elif partition.startswith("train") or partition.startswith('val'):
            #         # sample_partition = get_partition_age(vgg_identity, age_label)
            #         sample_partition = get_partition_identity(vgg_identity)
            #     label = age_label if only_age else (gender_category_label, age_label)

            if partition.startswith("test"):
                sample_partition = PARTITION_TEST
            elif partition.startswith("train") or partition.startswith('val'):
                # sample_partition = get_partition_age(vgg_identity, age_label)
                sample_partition = get_partition_identity(vgg_identity)
            else:
                print("Unkown partition", partition)
                exit(1)
            
            example = {
                'img': path,
                'label': age_label,
                'roi': get_roi(d[4:8]),
                'part': sample_partition
            }
            if np.max(img) == np.min(img):
                print('Warning, blank image: %s!' % path)
            else:
                data.append(example)
        else:
            print("WARNING! Unable to read %s" % path)
            n_discarded += 1
    print("Data loaded. %d samples (%d discarded)" % (len(data), n_discarded))
    return data




# def get_partition_age(identity_label, age_label):    
#     # try:
#     age_label = int(np.round(age_label))
#     return split_by_identity_age(age_label, identity_label)
#     # except ValueError:
#     #     return None


# def split_by_identity_age(age_label, identity_label):
#     global people_by_age
#     try:
#         faces, partition = people_by_age[age_label][identity_label]
#         people_by_age[age_label][identity_label] = (faces + 1, partition)
#     except ValueError:
#         # split 20/80 stratified by identity
#         l = (len(people_by_age[age_label]) - 1) % 10
#         if l == 0 or l == 1:
#             partition = PARTITION_VAL
#         else:
#             partition = PARTITION_TRAIN
#         people_by_age[age_label][identity_label] = (1, partition)
#     return partition


def get_partition_identity(identity_label):    
    global people_by_identity
    try:
        faces, partition = people_by_identity[identity_label]
        people_by_identity[identity_label] = (faces + 1, partition)
    except KeyError:
        # split 20/80 stratified by identity
        l = (len(people_by_identity) - 1) % 10
        if l == 0 or l == 1:
            partition = PARTITION_VAL
        else:
            partition = PARTITION_TRAIN
        people_by_identity[identity_label] = (1, partition)
    return partition

def print_verbose_partition(verbosed_partition):
    if verbosed_partition == PARTITION_TRAIN or verbosed_partition == PARTITION_VAL:
        train_identities, train_samples = 0, 0
        val_identities, val_samples = 0, 0

        print("Verbose partitions...")
        
        for _, (faces, partition) in people_by_identity.items():
            if partition == PARTITION_TRAIN:
                train_samples += faces
                train_identities += 1
            elif partition == PARTITION_VAL:
                val_samples += faces
                val_identities += 1

        train_identities_percentage = 100 * train_identities / (train_identities + val_identities)
        train_samples_percentage = 100 * train_samples / (train_samples + val_samples)

        print("Train identities {} ({}% of all identites)".format(train_identities, train_identities_percentage))
        print("Train samples {} ({}% of all identites)".format(train_samples, train_samples_percentage))

        val_identities_percentage = 100 * val_identities / (train_identities + val_identities)
        val_samples_percentage = 100 * val_samples / (train_samples + val_samples)

        print("validation identities {} ({}% of all identites)".format(val_identities, val_identities_percentage))
        print("Validation samples {} ({}% of all identites)".format(val_samples, val_samples_percentage))



class Vgg2DatasetAge(Vgg2Dataset):
    def __init__(self,
                 partition='train',
                 imagesdir='vggface2_data/<part>',
                 csvmeta='vggface2_data/annotations/<part>.detected.csv',
                 age_csv_meta='vggface2_data/annotations/<part>.age_detected.csv',
                 target_shape=(224, 224, 3),
                 augment=False,
                 custom_augmentation=None,
                 preprocessing='full_normalization',
                 debug_max_num_samples=None,
                 change_root_cached=False):
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

        # csv_meta_partition = csvmeta.replace('/', '_').replace('<part>', 'part')
        num_samples = "_"+str(debug_max_num_samples) if debug_max_num_samples is not None else ''

        cache_file_name = 'vggface2_age_{partition}{num_samples}.cache'.format(partition=partition, num_samples=num_samples)
        cache_file_name = os.path.join("dataset_cache", cache_file_name)
        cache_file_name = os.path.join(EXT_ROOT, cache_file_name)
        print("cache file name %s" % cache_file_name)
        try:
            with open(cache_file_name, 'rb') as f:
                self.data = pickle.load(f)[:debug_max_num_samples]
                if change_root_cached:
                    actual_root = os.path.dirname(imagesdir)
                    print("Changing dataset cached root path with %s ..." %
                          actual_root)
                    for x in tqdm(self.data):
                        subpath = os.path.relpath(
                            x['img'], x['img'].split("/")[0])
                        x['img'] = os.path.join(actual_root, subpath)
                print("Data loaded. %d samples, from cache" % (len(self.data)))
        except FileNotFoundError:
            print("Loading %s data from scratch" % partition)
            csvmeta = os.path.join(EXT_ROOT, csvmeta)
            age_csv_meta = os.path.join(EXT_ROOT, age_csv_meta)
            imagesdir = os.path.join(EXT_ROOT, imagesdir)
            load_partition = "train" if partition_label == PARTITION_TRAIN or partition_label == PARTITION_VAL else "test"
            loaded_data = _load_vgg2(csvmeta, imagesdir, load_partition, age_csv_meta, debug_max_num_samples)
            print_verbose_partition(verbosed_partition=partition_label)
            if partition.startswith('test'):
                self.data = loaded_data
            else:
                self.data = [x for x in loaded_data if x['part'] == partition_label]
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


def test_age(dataset="test", debug_samples=None):

    if dataset.startswith("train") or dataset.startswith("val"):
        print(dataset, debug_samples if debug_samples is not None else '')
        dt = Vgg2DatasetAge(dataset,
                            target_shape=(224, 224, 3),
                            preprocessing='vggface2',
                            debug_max_num_samples=debug_samples)
        print("SAMPLES %d" % dt.get_num_samples())


        # train_samples = list()
        # val_samples = list()
        # for _, identity_data in people_by_identity.items():
        #     if identity_data[1] == PARTITION_TRAIN:
        #         train_samples.append(identity_data[0])
        #     else:
        #         val_samples.append(identity_data[0])
        # print("Total train {} of different samples {}".format(sum(train_samples), len(train_samples)))
        # print("Total val {} of different samples {}".format(sum(val_samples), len(val_samples)))

        # print('Now generating from %s set' % dataset)
        gen = dt.get_generator()
    else:
        dv = Vgg2DatasetAge('test',
                            target_shape=(224, 224, 3),
                            preprocessing='vggface2',
                            debug_max_num_samples=debug_samples)
        # print("SAMPLES %d" % dv.get_num_samples())
        # print('Now generating from test set')
        gen = dv.get_generator()

    i = 0
    while True:
        i += 1
        for batch in tqdm(gen):
            for im, age in zip(batch[0], batch[1]):
                # age = np.argmax(age)
                facemax = np.max(im)
                facemin = np.min(im)

                print(im)

                im = (255 * ((im - facemin) / (facemax - facemin))).astype(np.uint8)
                cv2.putText(im, "{}".format(age), (0, im.shape[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255))
                cv2.imwrite('/tmp/vgg2im.jpg', im)
                cv2.imshow('vggface2 image', im)
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    return


if '__main__' == __name__:
    # test_age("train")
    # test_age("val")
    test_age("test")
