import argparse
import os
import re

results_by_model = {}
results_by_data = {}

parser = argparse.ArgumentParser(description='XLS conv script')
parser.add_argument('--input', dest='input_path', type=str, default='results.txt', help='text input file')
parser.add_argument('--output', dest="output_path", type=str, default='', help='XLS output file')
args = parser.parse_args()


def filter_model(txt):
    x = re.search("_net[0-9A-Za-z]+_", txt.split("/")[-2])
    return x.group()[4:-1] if x is not None else None


RESFORSTR = 'Results for: '
with open(args.input_path) as f:
    for line in f:
        if line.startswith(RESFORSTR):
            tgt_model = filter_model(line[len(RESFORSTR):].strip())
            results_by_model[tgt_model] = {}
        elif line.strip() == '':
            pass
        else:
            larr = line.split(' ')
            tgt_data = larr[0]
            res = float(larr[1])
            results_by_model[tgt_model][tgt_data] = res
            if not tgt_data in results_by_data.keys():
                results_by_data[tgt_data] = {}
            results_by_data[tgt_data][tgt_model] = res


def filter(txt):
    return os.path.basename(txt)


f = open(os.path.splitext(args.input_path)[0] + ".xls", 'w') if not args.output_path else open(args.output_path, 'w')


def out(txt, end='\n'):
    f.write(str(txt))
    f.write(end)


def sort(d):
    import collections
    d = collections.OrderedDict(sorted(d.items()))
    return d


header_printed = False
for data, modelres in sort(results_by_data).items():
    modelres = sort(modelres)
    if not header_printed:
        out('', end='\t')
        for model, res in modelres.items():
            out(filter(model), end='\t')
        out('')
        header_printed = True
    out(filter(data), end='\t')
    for model, res in modelres.items():
        out(res, end='\t')
    out('')

