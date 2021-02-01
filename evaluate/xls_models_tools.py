from collections import defaultdict


def filter_line(txt, escape='\t'):
    return txt.split(escape)


def remove_undesired(txt, escapes=('', '\t', '\n')):
    return [x for x in txt if x not in escapes]


def corruption_level(txt, escape='.'):
    entire = txt.split(escape)
    return (entire[-2], entire[-1]) if len(entire) > 1 else (entire[-1], None)


def extract_results(filepath, reveal_error=False):
    with open(filepath, 'r') as infile:
        data = infile.readlines()
        models = remove_undesired(filter_line(data[0]))
        results = {model: defaultdict(dict) for model in models}
        for i in data[1:]:
            values = filter_line(i)
            corruption, level = corruption_level(values[0])
            for model, val in zip(models, values[1:]):
                res = 1 - float(val) if reveal_error else float(val)
                if level is None:
                    results[model][corruption] = res
                else:
                    results[model][corruption][level] = res
    return results


def extract_results_by_corruption(filepath, reveal_error=False):
    results = defaultdict(dict)
    with open(filepath, 'r') as infile:
        data = infile.readlines()
        models = remove_undesired(filter_line(data[0]))
        for i in data[1:]:
            values = filter_line(i)
            corruption, level = corruption_level(values[0])
            for model, val in zip(models, values[1:]):
                res = 1 - float(val) if reveal_error else float(val)
                if level is None:
                    results[corruption][model] = res
                elif level not in results[corruption]:
                    results[corruption][level] = {model : res}
                else:
                    results[corruption][level][model] = res
    return results


def mean_dict(dict_values):
    return sum(dict_values.values()) / len(dict_values.values())