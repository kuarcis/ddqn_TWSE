import pandas
import os


data_dir = "./data/"

def xls_loader(dir = data_dir):

    print('select a xls file to read:')
    items = os.listdir(data_dir)
    fileList = [name for name in items if name.endswith(".xls")]

    for item in fileList:
        print('[',fileList.index(item),']:',item)

    load_mark = False
    while load_mark is not True:
        try:
            loadid= input('enter a number to read the file:[0-%d]' %(len(fileList)-1))
            loadid = int(loadid)
            if loadid >len(fileList) or loadid <0:
                raise Exception
            load_mark = True
            print('select [',loadid,']:', fileList[loadid],"....")
        except Exception as e:
            print('wrong type of input, try again')
            print(e)
    print('reading history price data....')
    stock_hp = pandas.read_excel((data_dir+fileList[loadid]),sheet_name='hp', header =0,index_col=0)

    stock_hp = stock_hp.fillna(method = 'ffill')

    stock_hp= stock_hp[stock_hp['Volume']!=0]


    print('hp read done')

    have_ca = False
    print('reading history CA data....')
    try:
        stock_ca = pandas.read_excel((data_dir+fileList[loadid]),sheet_name='ca')
        print('ca read done')
        have_ca = True
    except:
        print('no CA data read')

    try:
        return stock_hp, stock_ca, have_ca

    except:
        return stock_hp, None ,have_ca





