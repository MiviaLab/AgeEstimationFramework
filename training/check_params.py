import re
from glob import glob
import os
from keras.models import load_model
import sys

sys.path.insert(0, 'keras_vggface')
from keras_vggface.antialiasing import BlurPool
from mobile_net_v2_keras import relu6

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


dir_root = "out_training_fer"
params_file = "params_layers_nets_2.txt"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
VGGFACE_LOAD = True
MULTI_MODEL = True

for dir_net in glob(os.path.join(dir_root, '*')):
    max_ep, max_c = _find_latest_checkpoint(dir_net)
    model_name = max_c.split("/")[-2].split("_")[1][3:]
    print("counting %s..." % model_name)
    try:
        if model_name == "xception71" or model_name == "densenet121bc" or (model_name == "vgg16" and VGGFACE_LOAD):
            model = load_model(max_c, custom_objects={'BlurPool': BlurPool})
        elif model_name == "mobilenet64":
            model = load_model(max_c, custom_objects={'relu6': relu6})
        else:
            model = load_model(max_c)
        if MULTI_MODEL:
            model = model.layers[-2]
        model_params = model.count_params()
        model_layers = len(model.layers)
        print("params %s" % model_params)
        print("layers %d" % model_layers)
        with open(params_file, "a") as f:
            f.writelines(["%s\n" % model_name, "params: %s\n" % model_params, "layers: %d\n" % model_layers, "\n"])
    except ValueError as e:
        print(e)
