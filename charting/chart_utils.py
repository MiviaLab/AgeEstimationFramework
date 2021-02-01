import os
import re
import numpy as np
from glob import glob
from collections import defaultdict
import matplotlib.pyplot as plt

float_ex_re = r"[-+]?\d*\.\d+|\d+"

model_names = {
    "vgg16": "VGG-16",
    "senet50": "SE-ResNet-50",
    "densenet121bc": "DenseNet-121",
    "mobilenet224": "MobileNet v2-A",
    "mobilenet96": "MobileNet v2-B",
    "mobilenetv3large": "MobileNet v3-L",
    "mobilenetv3small": "MobileNet v3-S"
}

model_sort = list(model_names.values())

density = 3
patterns = ("||"*20, "", "++", "//", "||", "*", "o", "\\", ".", "-", "x", "O")
colors = ('crimson', 'royalblue', 'darkolivegreen', 'chocolate', 'lightseagreen', 'darkslateblue', 'sandybrown',
          'cornflowerblue', 'lightsalmon')
width = "XXXX"
    
def get_chart_path(input_path, output_path):
    results_name = os.path.basename(input_path[:-1] if input_path.endswith("/") else input_path)
    start_dataset = results_name.split("_")[1].upper()
    end_dataset = results_name.split("_")[3].upper()
    return os.path.join(output_path, "chart_trained_on_{}_and_tested_on_{}.png".format(start_dataset, end_dataset))

def get_net_name_from_summary_path(summary_path):
    summary_name = os.path.split(summary_path)[1]
    net_name = summary_name.split("_")[4][3:]
    return model_names[net_name] if net_name in model_names else net_name

def get_net_name_from_lap_summary_path(summary_path):
    summary_name = os.path.split(summary_path)[1]
    net_name = summary_name.split("_")[6][3:]
    return model_names[net_name] if net_name in model_names else net_name

def get_mae_from_summary(summary_path):
    with open(summary_path) as fp:
        text = fp.readlines()
    mae = re.findall(float_ex_re, text[2])
    assert "Mean absolute error" in text[2] and len(mae)==1, "Err deserializing {}\nAchieved: {}".format(summary_path, mae)
    return float(mae[0])

def get_epsscore_from_summary(summary_path):
    with open(summary_path) as fp:
        text = fp.readlines()
    if "Eps-score" in text[3]:
        eps_score = re.findall(float_ex_re, text[3])
    elif "Eps-score" in text[4]:
        eps_score = re.findall(float_ex_re, text[4])
    else:
        raise Exception("Eps-score not found in {}".format(summary_path))
    assert len(eps_score)==1, "Err deserializing {}\nAchieved: {}".format(summary_path, eps_score)
    return float(eps_score[0])

def get_score_from_tabulate(adience_file_path):
    fold_labels = ["fold_0", "fold_1", "fold_2", "fold_3", "fold_4", "fold_avg"]
    with open(adience_file_path) as fp:
        text = fp.readlines()
    scores = text[3].strip()
    scores = scores.split("|")[2:-1]
    scores = re.findall(float_ex_re, " ".join(scores))
    return {fold : float(fold_value) for fold, fold_value in zip(fold_labels, scores)}

def get_mae_from_original_and_corrupted(tab_path):
    with open(tab_path) as fp:
        text = fp.readlines()
    data = dict()
    for line in text[3::2]:
        model = line.split("|")[1].strip()
        original_mae = line.split("|")[2].strip()
        corrupted_mae = line.split("|")[3].strip()
        data[model] = [original_mae, corrupted_mae]
    return data

def get_mae_from_corruptions(tab_path):
    with open(tab_path) as fp:
        text = fp.readlines()
    data = dict()
    for line in text[3::2]:
        model = line.split("|")[1].strip()
        blur = line.split("|")[7].strip()
        digital = line.split("|")[10].strip()
        noise = line.split("|")[18].strip()
        data[model] = [blur, digital, noise]
    return data

def sort_data(data, sort=None):
    if sort is None:
        return data
    elif sort == "first_level":
        return {k:data[k] for k in sorted(data)}
    elif type(sort) is list:
        return {k:data[k] for k in sorted(data, key=lambda k:sort.index(k))}
    else:
        raise Exception("{} sort type not supperted.")

def get_data_mae_from_summary(input_files, sort=None):
    data = dict()
    for file_path in glob(input_files):
        if os.path.basename(file_path).startswith("summary_"):
            model_name = get_net_name_from_summary_path(file_path)
            model_data = get_mae_from_summary(file_path)
            data[model_name] = {"Mean Absolute Error" : model_data}
    return sort_data(data, sort)

def get_data_epsscore_from_summary(input_files, sort=None):
    data = dict()
    for file_path in glob(input_files):
        if os.path.basename(file_path).startswith("summary_"):
            model_name = get_net_name_from_lap_summary_path(file_path)
            model_data = get_epsscore_from_summary(file_path)
            data[model_name] = {"Epsilon score" : model_data}
    return sort_data(data, sort)

def get_data_adience_from_tabulate(input_files, sort=None):
    data = defaultdict(dict)
    top_one_string = "tabulate_adience_of_top_one_results"
    one_off_string = "tabulate_adience_of_one_off_results"

    for model_path in glob(input_files):
        model_dir_name = os.path.basename(model_path)
        if model_dir_name in model_names.keys():
            model_name = model_names[model_dir_name]
            complete_path = os.path.join(model_path, "*")
            # double cycle to keep top-one / one-off order
            for file_name in glob(complete_path):
                if os.path.basename(file_name).startswith(top_one_string):
                    model_data = get_score_from_tabulate(file_name)
                    data[model_name]["Top-one score"] = model_data
            for file_name in glob(complete_path):
                if os.path.basename(file_name).startswith(one_off_string):
                    model_data = get_score_from_tabulate(file_name)
                    data[model_name]["One-off score"] = model_data
    return data


def get_data_lfw_from_multi_tabulate(input_files, sort=None):
    data = defaultdict(dict)
    original_and_corrupted_tab = "tab1.txt"
    corruptions_tab = "tab2.txt"

    original_and_corrupted_avg = get_mae_from_original_and_corrupted(os.path.join(input_files, original_and_corrupted_tab))
    corruptions_avg = get_mae_from_corruptions(os.path.join(input_files, corruptions_tab))

    for model_name in original_and_corrupted_avg.keys():
        data[model_name]["LFW"] = original_and_corrupted_avg[model_name][0]
        data[model_name]["LFW+C"] = original_and_corrupted_avg[model_name][1]
        data[model_name]["Blur"] = corruptions_avg[model_name][0]
        data[model_name]["Noise"] = corruptions_avg[model_name][1] 
        data[model_name]["Digital"] = corruptions_avg[model_name][2]
    return data
        
    

def get_data_on_vggface2(input_files, sort=model_sort):
    return get_data_mae_from_summary(input_files, sort)

def get_data_on_imdbwiki(input_files, sort=model_sort):
    return get_data_mae_from_summary(input_files, sort)

def get_data_on_lap(input_files, sort=model_sort):
    return get_data_epsscore_from_summary(input_files, sort)

def get_data_on_adience_score(input_files, sort=model_sort):
    return get_data_adience_from_tabulate(input_files, sort)

def get_data_on_lfw_original_and_corrupted(input_files, sort=model_sort):
    return get_data_lfw_from_multi_tabulate(input_files, sort)

def get_data_on_lfw_corrupted_me(input_files):
    pass

# def create_chart_models(corruption_labels, models_dict, save_file_path='test.png', title='Accuracy by model',
#                         order_and_rename=True, official_labels=official_labels_lfw):
#     model_labels = list(models_dict.keys())
#     x = np.arange(len(model_labels))
#     width = 1.5 / (len(corruption_labels) + 4)
#     offset = (len(corruption_labels) - 2) / 2  # (len(corruption_labels) - 1) / 2
#     data_dict = defaultdict(list)
#     for i, corruption in enumerate(corruption_labels):
#         data_dict[corruption].extend([model_values[i] for model_values in models_dict.values()])
#     if order_and_rename:
#         data_dict = {official_labels[k]: v for k, v in data_dict.items()}

#         # order models by LFW+C
#         ordered_indexes = sorted(range(len(data_dict['LFW+C'])), key=lambda k: data_dict['LFW+C'][k], reverse=True)
#         data_dict = order_dict_of_list(dict_to_order=data_dict, ordered_indexes=ordered_indexes)
#         model_labels = [official_model_chart_refined[model_labels[i]] for i in ordered_indexes]
        
#         keyorder = {k: v for v, k in enumerate(official_labels.values())}
#         data_dict = OrderedDict(sorted(data_dict.items(), key=lambda i: keyorder.get(i[0])))

#     ncol = len(corruption_labels)
#     art = compile_chart(data_dict, width, title, x, offset, model_labels, ncol)#, special=1)
#     plt.savefig(save_file_path, additional_artists=art, bbox_inches="tight", dpi=300)

# def convert_chart_data(data):
#     models = list()
#     data_names = list()
#     data_values = list()
#     for model_name, model_data in data.items:
#         models.append(model_name)
#         if len(models) == 1 : 
#             data_names = list(model_data.keys())
#         data.values.append()

def pretty_data_print(d, indent=0):
   for key, value in d.items():
      print('\t' * indent + str(key))
      if isinstance(value, dict):
         pretty_data_print(value, indent+1)
      else:
         print('\t' * (indent+1) + str(value))            

def bar_chart(data, y_label, title=None, bounds=None, special=None):
    groups = list(data.keys())
    annotations = list(list(data.values())[0].keys())

    groups_range = np.arange(len(groups))
    bars_for_every_group = len(annotations)

    chart_data = list()
    for annotation in annotations:
        annotation_list = list()
        for group in groups:
            annotation_list.append(data[group][annotation])
        chart_data.append(annotation_list)


    fig, ax = plt.subplots()
    ax.set_title(title)
    plt.rcParams['hatch.linewidth'] = 1
    handler_legend = list()
    art = list()
    previous_color = None
    
    width = 1.5 / (bars_for_every_group + 4) #0.35

    for i, annotation_list in enumerate(chart_data):
        hatch = patterns[i] * density
        diff = width * (i - (bars_for_every_group-1)/2)
        label = annotations[i]
        if special is not None:
            if special == i:
                facecolor, edgecolor = "white", colors[i]
            else:
                facecolor, edgecolor = colors[i-1], colors[i-1]
        else:
            facecolor, edgecolor = colors[i], colors[i] 
        saved_ax = ax.bar(groups_range + diff, annotation_list, width, label=label, hatch=hatch, facecolor=facecolor,
                              edgecolor=edgecolor)
        handler_legend.append(saved_ax)


    # shift_index = 0
    # for i, (label, means) in enumerate(data.items()):
    #     if special is not None and i == special:
    #         # s_ax = ax.bar(x + diff, means, width - 0.02, label=label, facecolor=colors[i], edgecolor=colors[i])
    #         s_ax = ax.bar(x, [0] * len(means), width * len(data), bottom=means, label=label, fill=False,
    #                       edgecolor="black", linestyle='--', linewidth=0.7, zorder=3)
    #         handler_legend.append(Line2D([0], [0], color="black", linestyle='--', linewidth=0.7, label=label))
    #         shift_index = -1
    #     else:
    #         j = i + shift_index
    #         hatch = patterns[j] * density
    #         diff = width * (j - offset)
    #         if same_color is None:
    #             s_ax = ax.bar(x + diff, means, width - 0.02, label=label, hatch=hatch, facecolor="white",
    #                           edgecolor=colors[j])
    #         else:
    #             s_ax = ax.bar(x + diff, means, width - 0.02, label=label, hatch=hatch, facecolor=same_color[0],
    #                           edgecolor=same_color[1])
    #         handler_legend.append(s_ax)
    #     if upper_label:
    #         autolabel(ax, s_ax)

    if not y_label and len(annotations)==1:
        y_label = annotations[0]
    else:
        lgd = ax.legend(handles=handler_legend, loc=9, bbox_to_anchor=(0.5, -0.25), ncol=bars_for_every_group)
        art.append(lgd)

    ax.set_ylabel(y_label)
    ax.set_xticks(groups_range)
    ax.set_xticklabels(groups)

    print("Bounds:", bounds)

    if type(bounds) in [tuple, list]:
        if len(bounds) > 1:
            plt.ylim(bounds[0], bounds[1])
        if len(bounds) > 2:
            ax.set_yticks(np.arange(bounds[0], bounds[1], step=bounds[2]))
        if len(bounds) > 3:
            ax.set_yticks(np.arange(bounds[0], bounds[1], step=bounds[3]), minor=True)

    # And a corresponding grid
    ax.grid(which='both')
    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.1)
    ax.grid(which='major', alpha=0.3)

    fig.autofmt_xdate()
    return art


def save_bar_chart(output_path, data, y_label="", title=None, bounds=None, special=None):
    art = bar_chart(data, y_label, title=title, bounds=bounds, special=special)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, additional_artists=art, bbox_inches="tight", dpi=300)