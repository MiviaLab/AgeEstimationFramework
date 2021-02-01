import argparse
import os
from chart_utils import get_chart_path, get_data_on_adience_score, pretty_data_print, save_bar_chart
from xlsx_utils import get_xlsx_path, write_xlsx
from collections import defaultdict

parser = argparse.ArgumentParser(description='Adience top-1 and 1-off chart')
parser.add_argument('--input', dest='input', required=True, type=str, help='input directory containing results to chart')
parser.add_argument('--output', dest='output', type=str, default="charts", help='output directory in which to chart')
parser.add_argument('--title', dest='title', type=str, default="", help='title of the chart')
parser.add_argument('--bounds', dest='bounds', type=str, help='Y-bounds lower:upper[:step[:minor_step]]')
args = parser.parse_args()


input_files = os.path.join(args.input, "*")
output_chart = get_chart_path(args.input, args.output)
chart_bounds = None if args.bounds is None else tuple([float(x) for x in args.bounds.split(":")])
output_xlsx = get_xlsx_path(args.input, args.output)

data = get_data_on_adience_score(input_files=input_files)

# Combined top-one and one-off avgs
print("Elaborating combined chart...")
avg_data = defaultdict(dict)
for model_name, model_data in data.items():
    for score_type, folds_dict in model_data.items():
        avg_data[model_name][score_type] = folds_dict['fold_avg']

pretty_data_print(avg_data)

print("Saving chart...")
save_bar_chart(output_chart, avg_data, y_label="", title=args.title, bounds=chart_bounds)
print("Chart {} saved!".format(output_chart))

print("Saving XLSX...")
write_xlsx(output_xlsx, avg_data)
print("XLSX {} saved!".format(output_xlsx))

# Only top-one avg
print("Elaborating Top-1 chart...")
top_one_data = defaultdict(dict)
for model_name, model_data in data.items():
    top_one_data[model_name]["Top-one score"] = model_data["Top-one score"]['fold_avg']

pretty_data_print(top_one_data)

print("Saving chart...")
top_one_output_chart = os.path.splitext(output_chart)[0] + "_top_one" + os.path.splitext(output_chart)[1]
save_bar_chart(top_one_output_chart, top_one_data, y_label="", title=args.title, bounds=chart_bounds)
print("Chart {} saved!".format(top_one_output_chart))

print("Saving XLSX...")
top_one_output_xlsx = os.path.splitext(output_xlsx)[0] + "_top_one" + os.path.splitext(output_xlsx)[1]
write_xlsx(top_one_output_xlsx, top_one_data)
print("XLSX {} saved!".format(top_one_output_xlsx))

# Only one-off avg
print("Elaborating 1-off chart...")
one_off_data = defaultdict(dict)
for model_name, model_data in data.items():
    one_off_data[model_name]["One-off score"] = model_data["One-off score"]['fold_avg']

pretty_data_print(one_off_data)

print("Saving chart...")
one_off_output_chart = os.path.splitext(output_chart)[0] + "_one_off" + os.path.splitext(output_chart)[1]
save_bar_chart(one_off_output_chart, one_off_data, y_label="", title=args.title, bounds=chart_bounds)
print("Chart {} saved!".format(one_off_output_chart))

print("Saving XLSX...")
one_off_output_xlsx = os.path.splitext(output_xlsx)[0] + "_one_off" + os.path.splitext(output_xlsx)[1]
write_xlsx(one_off_output_xlsx, one_off_data)
print("XLSX {} saved!".format(one_off_output_xlsx))