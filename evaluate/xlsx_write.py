import xlsxwriter
import sys
import os

#PARAMETERS
COLUMNS_WITDH = 10
TEXT_ALIGNMENT = 'center'
TABLE_STYLE = 'Table Style Medium 4'

def detabulate(txt_path):

    import re 
  
    def remove(string): 
        pattern = re.compile(r'\s+') 
        return re.sub(pattern, '', string)

    matrix = []
    with open(txt_path,'r') as f:
        for row in f.readlines():
            if not row.startswith('+'):
                matrix.append([ remove(x) for x in row.split('|') if len(remove(x)) > 0 ])

    return matrix

def write_xlsx(filename,data):

    workbook = xlsxwriter.Workbook(filename)

    center_format = workbook.add_format()
    center_format.set_align(TEXT_ALIGNMENT)

    worksheet = workbook.add_worksheet()
    
    worksheet.set_column(0,0,COLUMNS_WITDH+5)
    worksheet.set_column(1,len(data[0])-1,COLUMNS_WITDH)

    columns = [ {'header':x,'format':center_format} for x in data[0] ]
    
    last_cell_index = xlsxwriter.utility.xl_range(0,0,len(data),len(data[0])-1)
    worksheet.add_table(last_cell_index,{'header_row': 0, 'data':data, 'style': TABLE_STYLE, 'columns': columns})
    
    workbook.close()

if __name__ == '__main__':
    data = detabulate(sys.argv[1])
    output_filename = os.path.splitext(sys.argv[1])[0] + ".xlsx"
    print("Writing on", output_filename)
    write_xlsx(output_filename,data)