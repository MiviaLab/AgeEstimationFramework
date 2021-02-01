import argparse
from xls_models_tools import mean_dict, extract_results_by_corruption
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import defaultdict, OrderedDict
from tabulate import tabulate
from matplotlib.lines import Line2D

from lfw_plot_accuracy_from_xls import average_by_levels, nine_models_order, compile_chart

parser = argparse.ArgumentParser(description='corruption error calculation')
parser.add_argument('--corrupted', dest='corrupted', type=str, help='corrupted experiment results (.xls)')
parser.add_argument('--uncorrupted', dest='uncorrupted', type=str, help='original experiment results (.xls)')
parser.add_argument('--out', dest='filepath', type=str, help='output file path of the chart')
parser.add_argument('--title', dest='title', type=str, help='title of the chart')
args = parser.parse_args()

VGGFace2_train = {
    "vgg16": 0.9998,
    "senet50": 0.9893,
    "densenet121bc": 0.9895,
    "mobilenet224": 0.9892,
    "mobilenet96": 0.9829,
    "mobilenet64": 0.9724,
    "shufflenet224": 0.9864,
    "squeezenet": 0.9593,
    "xception71": 0.9937,
}

VGGFace2_val = {
    "vgg16": 0.9836,
    "senet50": 0.9825,
    "densenet121bc": 0.9823,
    "mobilenet224": 0.9814,
    "mobilenet96": 0.9795,
    "mobilenet64": 0.9728,
    "shufflenet224": 0.9821,
    "squeezenet": 0.9648,
    "xception71": 0.9796,
}

official_labels_vgg_lfw = {
    "vgg-train": "VGGFace2 train",
    "vgg-val": "VGGFace2 validation",
    "LFW": "LFW+",
    "LFW-C": "LFW+C",
}

fantasy = ("*", "..", "xx", "\\", "++", "//", "||", "--", "o", "O")
colors = ("darkcyan", "mediumorchid", "crimson", "black", 'lightseagreen', 'darkslateblue', 'sandybrown',
          'cornflowerblue', 'lightsalmon', 'royalblue', 'darkolivegreen', 'chocolate')


def create_chart_models(data_labels, models_dict, save_file_path='test.png', title=''):
    model_labels = list(models_dict.keys())
    x = np.arange(len(model_labels))
    width = 1.5 / (len(data_labels) + 4)
    offset = (len(data_labels) - 1) / 2  # (len(corruption_labels) - 1) / 2
    data_dict = defaultdict(list)
    for i, lab in enumerate(data_labels):
        data_dict[lab].extend([model_values[i] for model_values in models_dict.values()])
    # if order_and_rename:
    #     data_dict = {official_labels[k]: v for k, v in data_dict.items()}
    #     keyorder = {k: v for v, k in enumerate(official_labels.values())}
    #     data_dict = OrderedDict(sorted(data_dict.items(), key=lambda i: keyorder.get(i[0])))
    ncol = len(data_labels)
    art = compile_chart(data_dict, width, title, x, offset, model_labels, ncol, patterns=fantasy, colors=colors)
    # same_color=('white', 'indianred'))
    plt.savefig(save_file_path, additional_artists=art, bbox_inches="tight", dpi=300)


def plot_bar_chart(corrupted_exp, uncorrupted_exp, filepath, title, by_category=False, debug=False):
    data_means = defaultdict(dict)
    data_means_compress = defaultdict(list)
    data_means_corrupt_dict = defaultdict(dict)

    corruptions = list(official_labels_vgg_lfw.values())

    data_means = average_by_levels(corrupted_exp, data_means)

    # append vgg-train
    for model, vgg_train in VGGFace2_train.items():
        data_means_compress[model].append(vgg_train)

    # append vgg-val
    for model, vgg_val, in VGGFace2_val.items():
        data_means_compress[model].append(vgg_val)

    # append LFW
    uncorr_data = next(iter(uncorrupted_exp.values()))
    for model, uncorr_value in uncorr_data.items():
        data_means_compress[model].append(uncorr_value)

    # append LFW-C
    for model, corr_dict in data_means.items():
        tmp_list = list()
        for corr_key, corr_mean in corr_dict.items():
            tmp_list.append(corr_mean)
        data_means_compress[model].append(sum(tmp_list) / len(tmp_list))

    data_means_compress = nine_models_order(data_means_compress)

    create_chart_models(corruptions, data_means_compress, filepath, title)


if __name__ == '__main__':
    corrupted_results = extract_results_by_corruption(args.corrupted)
    uncorrupted_results = extract_results_by_corruption(args.uncorrupted)
    plot_bar_chart(corrupted_results, uncorrupted_results, args.filepath, args.title)
