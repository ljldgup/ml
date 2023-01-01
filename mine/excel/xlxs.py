import pandas as pd

path1 = r'D:\exp\temp\402营业成本-合并——抽凭.xlsx'
path2 = r'D:\exp\temp\成本费用抽凭表.xls'
if __name__ == '__main__':
    data1 = pd.read_excel(path1, head=4, sheet_name=1)
    data2 = pd.read_excel(path2, head=0, sheet_name=1)

    print(data2[data2['核算账簿'] == data2['核算账簿'].unique()[1]][['核算账簿','索引号']])