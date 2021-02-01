import argparse
from xls_models_tools import mean_dict, extract_results_by_corruption
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import defaultdict, OrderedDict
from tabulate import tabulate
from matplotlib.lines import Line2D

parser = argparse.ArgumentParser(description='corruption error calculation')
parser.add_argument('--corrupted', dest='corrupted', type=str, help='corrupted experiment results (.xls)')
parser.add_argument('--uncorrupted', dest='uncorrupted', type=str, help='original experiment results (.xls)')
parser.add_argument('--out', dest='filepath', type=str, help='output file path of the chart')
parser.add_argument('--title', dest='title', type=str, default="", help='title of the chart') # Performance on the original and corrupted dataset
args = parser.parse_args()

allowed_tails = ['', '_minus', '_plus']
UNCORRUPTED_LABEL = "LFW"
AVERAGE_CORRUPTION_LABEL = "LFW-C"
density = 3
fantasy = ("", "++", "//", "||", "*", "o", "\\", ".", "-", "x", "O")
fantasy_refined = ("", "|"*20, "++", "//", "||", "*", "o", "\\", ".", "-", "x", "O")

colors = ('crimson', 'royalblue', 'darkolivegreen', 'chocolate', 'lightseagreen', 'darkslateblue', 'sandybrown',
          'cornflowerblue', 'lightsalmon')
colors_refined = ('crimson', 'crimson', 'royalblue', 'darkolivegreen', 'chocolate', 'lightseagreen', 'darkslateblue', 'sandybrown',
          'cornflowerblue', 'lightsalmon')

noise_labels = ['gaussian_noise', 'shot_noise']
blur_labels = ['defocus_blur', 'gaussian_blur', 'motion_blur', 'zoom_blur']
digital_labels = ['brightness_minus', 'brightness_plus', 'contrast', 'contrast_plus', 'jpeg_compression', 'pixelate', 'spatter']
LATEX_TAB = False

official_model = {
    "vgg16": "VGG-16",
    "senet50": "SE-ResNet-50",
    "densenet121bc": "DenseNet-121",
    "mobilenet224": "MobileNet v2-A",
    "mobilenet96": "MobileNet v2-B",
    "mobilenet64": "MobileNet v2-C",
    "shufflenet224": "ShuffleNet",
    "squeezenet": "SqueezeNet",
    "xception71": "XceptionNet",
    "mobilenetv3small" : "Mobilenet v3-S",
    "mobilenetv3large" : "Mobilenet v3-L"
}

official_model_chart_refined = {
    "VGG-16" : "VGG",
    "SE-ResNet-50" : "SENet",
    "DenseNet-121" : "DenseNet",
    "MobileNet v2-A" : "MobileNet-A",
    "MobileNet v2-B" : "MobileNet-B",
    "MobileNet v2-C" : "MobileNet-C",
    "ShuffleNet" : "ShuffleNet",
    "SqueezeNet" : "SqueezeNet",
    "XceptionNet" : "Xception",
    "Mobilenet v3-S" : "Mobilenet v3-S",
    "Mobilenet v3-L" : "Mobilenet v3-L"
}

official_labels_lfw = {
    "LFW": "LFW+",
    "LFW-C": "LFW+C",
    "blur": "Blur",
    "noise": "Noise",
    "digital": "Digital"
}

LOWER_BOUND_CHART = 0.5

def corruption_category_combination(corruption):
    if any(c in corruption for c in ['gaussian', 'shot']):
        return 'noise'
    elif any(c in corruption for c in ['defocus', 'motion', 'zoom']):
        return 'blur'
    elif any(c in corruption for c in ['brightness', 'contrast', 'jpeg', 'pixel', 'spatter']):
        return 'digital'
    else:
        return corruption


def corruption_category(corruption):
    if corruption in noise_labels:
        return 'noise'
    elif corruption in blur_labels:
        return 'blur'
    elif corruption in digital_labels:
        return 'digital'
    else:
        return None

def order_dict_of_list(dict_to_order, ordered_indexes):
    ordered = dict()
    for k, vlist in dict_to_order.items():
        ordered[k] = [vlist[i] for i in ordered_indexes]
    return ordered


def autolabel(ax, rects):
    for rect in rects:
        height = rect.get_height()
        fontsize = rect.get_width() * 40
        ax.annotate('{}'.format(round(height, 2)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 2),
                    textcoords="offset points",
                    fontsize=fontsize,
                    ha='center', va='bottom', rotation=90)


def create_chart_category(corruption_labels, models_dict, save_file_path='test.png', title='Accuracy by corruption'):
    x = np.arange(len(corruption_labels))
    width = 1.5 / (len(models_dict) + 4)
    offset = (len(models_dict) - 1) / 2
    ncol = len(models_dict) / 2
    art = compile_chart(models_dict, width, title, x, offset, corruption_labels, ncol)
    plt.savefig(save_file_path, additional_artists=art, bbox_inches="tight", dpi=300)


def create_chart_models(corruption_labels, models_dict, save_file_path='test.png', title='Accuracy by model',
                        order_and_rename=True, official_labels=official_labels_lfw):
    model_labels = list(models_dict.keys())
    x = np.arange(len(model_labels))
    width = 1.5 / (len(corruption_labels) + 4)
    offset = (len(corruption_labels) - 2) / 2  # (len(corruption_labels) - 1) / 2
    data_dict = defaultdict(list)
    for i, corruption in enumerate(corruption_labels):
        data_dict[corruption].extend([model_values[i] for model_values in models_dict.values()])
    if order_and_rename:
        data_dict = {official_labels[k]: v for k, v in data_dict.items()}

        # order models by LFW+C
        ordered_indexes = sorted(range(len(data_dict['LFW+C'])), key=lambda k: data_dict['LFW+C'][k], reverse=True)
        data_dict = order_dict_of_list(dict_to_order=data_dict, ordered_indexes=ordered_indexes)
        model_labels = [official_model_chart_refined[model_labels[i]] for i in ordered_indexes]
        
        keyorder = {k: v for v, k in enumerate(official_labels.values())}
        data_dict = OrderedDict(sorted(data_dict.items(), key=lambda i: keyorder.get(i[0])))

    ncol = len(corruption_labels)
    art = compile_chart(data_dict, width, title, x, offset, model_labels, ncol)#, special=1)
    plt.savefig(save_file_path, additional_artists=art, bbox_inches="tight", dpi=300)


def compile_chart(data, width, title, x, offset, tick_labels, ncol, special=None, upper_label=False, patterns=fantasy_refined,
                  colors=colors_refined, same_color=None):
    fig, ax = plt.subplots()
    ax.set_title(title)
    plt.rcParams['hatch.linewidth'] = 1
    handler_legend = list()
    shift_index = 0
    for i, (label, means) in enumerate(data.items()):
        if special is not None and i == special:
            # s_ax = ax.bar(x + diff, means, width - 0.02, label=label, facecolor=colors[i], edgecolor=colors[i])
            s_ax = ax.bar(x, [0] * len(means), width * len(data), bottom=means, label=label, fill=False,
                          edgecolor="black", linestyle='--', linewidth=0.7, zorder=3)
            handler_legend.append(Line2D([0], [0], color="black", linestyle='--', linewidth=0.7, label=label))
            shift_index = -1
        else:
            j = i + shift_index
            hatch = patterns[j] * density
            diff = width * (j - offset)
            if same_color is None:
                s_ax = ax.bar(x + diff, means, width - 0.02, label=label, hatch=hatch, facecolor="white",
                              edgecolor=colors[j])
            else:
                s_ax = ax.bar(x + diff, means, width - 0.02, label=label, hatch=hatch, facecolor=same_color[0],
                              edgecolor=same_color[1])
            handler_legend.append(s_ax)
        if upper_label:
            autolabel(ax, s_ax)
    ax.set_ylabel('Accuracy')
    ax.set_xticks(x)

    ax.set_xticklabels(tick_labels)
    art = []
    lgd = ax.legend(handles=handler_legend, loc=9, bbox_to_anchor=(0.5, -0.25), ncol=ncol)
    art.append(lgd)
    plt.ylim(0.8, 1)

    ax.set_yticks(np.arange(LOWER_BOUND_CHART, 1.01, step=0.1))
    ax.set_yticks(np.arange(LOWER_BOUND_CHART, 1.001, step=0.01), minor=True)

    # And a corresponding grid
    ax.grid(which='both')
    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.1)
    ax.grid(which='major', alpha=0.3)

    fig.autofmt_xdate()
    return art


def nine_models_order(data, rename=True):
    if rename:
        data = {official_model[k]: v for k, v in data.items()}
    keyorder = {k: v for v, k in enumerate(official_model.values() if rename else official_model.keys())}
    return OrderedDict(sorted(data.items(), key=lambda i: keyorder.get(i[0])))


def average_by_levels(corrupted_exp, data_means):
    for corr, corr_data in corrupted_exp.items():
        tmp = defaultdict(list)
        for level, model_data in corr_data.items():
            for model, data in model_data.items():
                tmp[model].append(data)
        for model, data in tmp.items():
            mean = sum(data) / len(data)
            data_means[model][corr] = mean
    return data_means


def category_compression(data_means, data_means_compress, data_means_corrupt_dict, corruptions, debug):
    for model, corr_dict in data_means.items():
        tmp = defaultdict(list)
        for corr_key, value in corr_dict.items():
            corruption_category_checked = corruption_category(corr_key)
            if debug:
                print(corr_key, corruption_category_checked)
            if corruption_category_checked is not None:
                tmp[corruption_category_checked].append(value)
        for corr_cat, data in tmp.items():
            mean = sum(data) / len(data)
            data_means_corrupt_dict[model][corr_cat] = mean
            data_means_compress[model].append(mean)
        if corruptions is None:
            corruptions = list(tmp.keys())
    return data_means_compress, data_means_corrupt_dict, corruptions


def plot_bar_chart(corrupted_exp, uncorrupted_exp, filepath, title, by_category=False, debug=False):
    """
    # corrupted_exp = {
    #   corruption : {
    #       level : {
    #           model : value,
    #           ... : ...
    #           },
    #       ... : ...
    #       },
    #   ... : ...
    #   }

    # uncorrupted_exp = {
    #   sample_label : {
    #       model : value,
    #       ... : ...
    #       }
    #   }
    """

    data_means = defaultdict(dict)
    data_means_compress = defaultdict(list)
    data_means_corrupt_dict = defaultdict(dict)
    corruptions = None

    # corrupted experiment results averaged by levels
    # for corr, corr_data in corrupted_exp.items():
    #     tmp = defaultdict(list)
    #     for level, model_data in corr_data.items():
    #         for model, data in model_data.items():
    #             tmp[model].append(data)
    #     for model, data in tmp.items():
    #         mean = sum(data) / len(data)
    #         data_means[model][corr] = mean
    data_means = average_by_levels(corrupted_exp, data_means)

    """
    # data_means = {
    #   model : {
    #       corruption : average value on levels,
    #       ... : ...
    #       },
    #   ... : ...
    #   }
    """

    # category compression
    # for model, corr_dict in data_means.items():
    #     tmp = defaultdict(list)
    #     for corr_key, value in corr_dict.items():
    #         corruption_category_checked = corruption_category(corr_key)
    #         if debug:
    #             print(corr_key, corruption_category_checked)
    #         if corruption_category_checked is not None:
    #             tmp[corruption_category_checked].append(value)
    #     for corr_cat, data in tmp.items():
    #         mean = sum(data) / len(data)
    #         data_means_corrupt_dict[model][corr_cat] = mean
    #         data_means_compress[model].append(mean)
    #     if corruptions is None:
    #         corruptions = list(tmp.keys())
    data_means_compress, data_means_corrupt_dict, corruptions = category_compression(data_means,
                                                                                     data_means_compress,
                                                                                     data_means_corrupt_dict,
                                                                                     corruptions, debug)

    # insert in list head uncorrupted experiment results
    uncorr_data = next(iter(uncorrupted_exp.values()))
    for model, uncorr_value in uncorr_data.items():
        data_means_compress[model].insert(0, uncorr_value)
    corruptions.insert(0, UNCORRUPTED_LABEL)

    # insert in list head average corrupted experiment results
    for model, corr_dict in data_means.items():
        tmp_list = list()
        for corr_key, corr_mean in corr_dict.items():
            if corruption_category(corr_key) is not None:
                tmp_list.append(corr_mean)
        data_means_compress[model].append(sum(tmp_list) / len(tmp_list))
    corruptions.append(AVERAGE_CORRUPTION_LABEL)

    """
    # data_means_compress = {
    #   model : [cat1_corr_val, cat2_corr_val, ..., uncorrupted_exp_val],
    #   ... : ...
    #   }
    """
    data_means_compress = nine_models_order(data_means_compress)
    uncorr_data = nine_models_order(uncorr_data)
    data_means = nine_models_order(data_means)
    data_means_corrupt_dict = nine_models_order(data_means_corrupt_dict)

    # create chart grouping by corruption categories or by models
    if by_category:
        create_chart_category(corruptions, data_means_compress, os.path.join(filepath, "category_chart"), title)
    else:
        create_chart_models(corruptions, data_means_compress, os.path.join(filepath, "models_chart"), title)

    ##### SMALL TABULATE TO FILE #####
    col_labels = ['Method', 'LFW+', 'LFW+C']
    row_labels = [m for m in uncorr_data.keys()]
    lfw_values = [v for v in uncorr_data.values()]
    # lfw_c_values = [sum(corr_dict.values()) / len(corr_dict.values()) for corr_dict in data_means.values()]

    lfw_c_values = list()
    for model, corr_dict in data_means.items():
        tmp = list()
        for corr_key, corr_val in corr_dict.items():
            if corr_key in blur_labels or corr_key in noise_labels or corr_key in digital_labels:
                tmp.append(corr_val)
        lfw_c_values.append(sum(tmp)/len(tmp))
    del tmp


    table_vals = [[r, round(v, 3), round(c, 3)] for r, v, c in zip(row_labels, lfw_values, lfw_c_values)]

    tab_1 = tabulate(table_vals, headers=col_labels, tablefmt="latex" if LATEX_TAB else "grid")

    print(tab_1)
    with open(os.path.join(filepath, "tab1.txt"), 'w') as f:
        f.write(tab_1)

    ##### LARGE TABULATE TO FILE #####
    col_labels = ['Method', 'LFW+C']
    col_labels.extend(blur_labels)
    col_labels.append('blur Avg')
    col_labels.extend(noise_labels)
    col_labels.append('noise Avg')
    col_labels.extend(digital_labels)
    col_labels.append('digital Avg')

    table_vals = list()

    for model, corr_data in data_means.items():
        # tmp = [model, sum(corr_data.values()) / len(corr_data.values())]

        tmp_complete_average = 0
        tmp = [model]

        for l in blur_labels:
            tmp.append(corr_data[l])
            tmp_complete_average += corr_data[l]
        tmp.append(data_means_corrupt_dict[model]['blur'])
        for l in noise_labels:
            tmp.append(corr_data[l])
            tmp_complete_average += corr_data[l]
        tmp.append(data_means_corrupt_dict[model]['noise'])
        for l in digital_labels:
            tmp.append(corr_data[l])
            tmp_complete_average += corr_data[l]
        tmp.append(data_means_corrupt_dict[model]['digital'])

        tmp_complete_average = tmp_complete_average / (len(blur_labels) + len(noise_labels) + len(digital_labels))
        tmp.insert(1,tmp_complete_average)

        table_vals.append(tmp)

    tab_2 = tabulate(table_vals, headers=col_labels, tablefmt="latex" if LATEX_TAB else "grid")

    print(tab_2)
    with open(os.path.join(filepath, "tab2.txt"), 'w') as f:
        f.write(tab_2)

    if debug:
        import json
        print(json.dumps(corrupted_results, indent=1))
        print(json.dumps(uncorrupted_results, indent=1))
        print(json.dumps(data_means_compress, indent=1))


if __name__ == '__main__':
    corrupted_results = extract_results_by_corruption(args.corrupted)
    uncorrupted_results = extract_results_by_corruption(args.uncorrupted)
    plot_bar_chart(corrupted_results, uncorrupted_results, args.filepath, args.title)