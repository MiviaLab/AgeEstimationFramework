import argparse
import os
from chart_utils import get_chart_path, get_data_on_vggface2, pretty_data_print, save_bar_chart
from xlsx_utils import get_xlsx_path, write_xlsx

parser = argparse.ArgumentParser(description='VGGFace2 Age chart')
parser.add_argument('--input', dest='input', required=True, type=str, help='input directory containing results to chart')
parser.add_argument('--output', dest='output', type=str, default="charts", help='output directory in which to chart')
parser.add_argument('--title', dest='title', type=str, default="", help='title of the chart')
parser.add_argument('--bounds', dest='bounds', type=str, help='Y-bounds lower:upper[:step[:minor_step]]')
args = parser.parse_args()


input_files = os.path.join(args.input, "*")
output_chart = get_chart_path(args.input, args.output)
chart_bounds = None if args.bounds is None else tuple([float(x) for x in args.bounds.split(":")])
output_xlsx = get_xlsx_path(args.input, args.output)

data = get_data_on_vggface2(input_files=input_files)

pretty_data_print(data)

print("Saving chart...")
save_bar_chart(output_chart, data, y_label="", title=args.title, bounds=chart_bounds)
print("Chart {} saved!".format(output_chart))

print("Saving XLSX...")
write_xlsx(output_xlsx, data)
print("XLSX {} saved!".format(output_xlsx))

