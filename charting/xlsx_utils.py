import xlsxwriter
import sys
import os

TEXT_ALIGNMENT = 'center'
TABLE_STYLE = 'Table Style Medium 4'

def from_nested_dicts_to_xlsx_values(nested_dicts):
    column_labels = list(next(iter(nested_dicts.values())).keys())
    values = list()
    for first_value, nested in nested_dicts.items():
        tmp = [first_value] + [v for v in nested.values()]
        values.append(tmp)
    return column_labels, values, len(values[0]), len(values)


def write_xlsx(filename,data):
    columns_labels, values, num_of_columns, num_of_rows = from_nested_dicts_to_xlsx_values(data)

    workbook = xlsxwriter.Workbook(filename)
    header_format = workbook.add_format({
            'align': 'center',
            'text_wrap': True
            })

    columns_labels.insert(0, "Method")
    columns = [{'header':label,'format':header_format} for label in columns_labels]
    
    last_cell_index = xlsxwriter.utility.xl_range(0, 0, num_of_rows, num_of_columns-1)
    values.insert(0, columns_labels)

    max_column_width = 0
    for sub_list in values:
        for element in sub_list:
            if len(str(element)) > max_column_width:
                max_column_width = len(str(element))

    worksheet = workbook.add_worksheet()
    worksheet.add_table(last_cell_index,{'header_row': 0, 'data': values, 'style': TABLE_STYLE, 'columns': columns})
    worksheet.set_column(0, num_of_columns, max_column_width)
    
    workbook.close()


def get_xlsx_path(input_path, output_path):
    results_name = os.path.basename(input_path[:-1] if input_path.endswith("/") else input_path)
    start_dataset = results_name.split("_")[1].upper()
    end_dataset = results_name.split("_")[3].upper()
    return os.path.join(output_path, "chart_trained_on_{}_and_tested_on_{}.xlsx".format(start_dataset, end_dataset))