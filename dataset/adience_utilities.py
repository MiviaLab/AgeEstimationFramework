import re
import numpy as np
import ast
from collections import defaultdict
import os

FEMALE_LABEL = 0
MALE_LABEL = 1

AGE_CLASSES = ((0, 2), (4, 6), (8, 12), (15, 20), (25, 32), (38, 43), (48, 53), (60, 100))

IMAGE_PATH = "<images_dir>/faces/<user_id>/coarse_tilt_aligned_face.<face_id>.<original_image>"
ANNOTATION_PATH = "<images_dir>/annotations/fold_frontal_<fold_number>_data.txt"
LANDMARKS_PATH = "<images_dir>/faces/<user_id>/landmarks.<face_id>.<original_image_id>.txt"

available_folds = (0, 1, 2, 3, 4)

def _get_age_label(floating_string, precision=3):
    return float(floating_string) if precision is None else np.round(float(floating_string), precision)


def _get_gender_label(label):
    if label.startswith("m"):
        return MALE_LABEL
    elif label.startswith("f"):
        return FEMALE_LABEL
    return None


def get_metafold(fold_label):
    try:
        fold = re.match(r"^fold_\d+$", fold_label).group(0)
    except AttributeError:
        raise Exception('Fold label must be: "fold_" followed by number')
    foldnumber = int(fold[5:])
    if foldnumber not in available_folds:
        raise Exception('Invalid fold number {}'.format(foldnumber))
    return foldnumber


def _format_adience_value(value):
    try:
        # None, (number, number), number
        return ast.literal_eval(value)
    except ValueError:
        return value

def _valid_line(line, expected_length):
    x = _format_adience_value(line[3])
    if type(x) is tuple and x not in AGE_CLASSES:
        print("Not in age classes:", x)
    return '' not in line and len(line) == expected_length and _format_adience_value(line[3]) in AGE_CLASSES


def _get_age_index(stringvalue):
    tuplevalue = _format_adience_value(stringvalue)
    return AGE_CLASSES.index(tuplevalue)


def get_structured_adience_meta(images_path, metatxt, fold_number, landmarks):
    with open(metatxt) as f:
        meta = f.read().splitlines()

    header = meta[0].split("\t")
    data = defaultdict(dict)
    discarded = 0

    for line in meta[1:]:
        line = line.split("\t")
        if _valid_line(line, len(header)):
            key = images_path.replace("<user_id>", line[0]).replace("<face_id>", line[2]).replace("<original_image>", line[1])
            data[key] = {head: _format_adience_value(value) for head, value in zip(header[4:], line[4:])}
            data[key]["age_label"] = _get_age_index(line[3])
            tmp_landmarks = landmarks.replace("<user_id>", line[0]).replace("<face_id>", line[2])
            data[key]['landmarks'] = tmp_landmarks.replace("<original_image_id>", os.path.splitext(line[1])[0])
        else:
            discarded += 1
    print("Loading meta: {} entries acquired, {} discarded for incompleteness".format(len(data), discarded))
    return data


def get_roi(d):
    return [int(d[0]), int(d[1]), int(d[2]), int(d[3])]

def get_roi_from_landmarks(landmarks, width, height):
    minX, minY = width, height
    maxX, maxY = 0, 0
    for point in landmarks:
        if point[0] < minX:
            minX = point[0]
        elif point[0] > maxX:
            maxX = point[0]
        elif point[1] < minY:
            minY = point[1]
        elif point[1] > maxY:
            maxY = point[1]

    return [minX, minY, maxX - minX, maxY - minY]

def parse_landmarks(landmarks_path):
    with open(landmarks_path) as f:
        lands = f.read().splitlines()
    correct_lands = list()
    for l in lands[2:]:
        l = l.split(',')
        x, y = int(_format_adience_value(l[-2])), int(_format_adience_value(l[-1]))
        correct_lands.append((x, y))
    return correct_lands