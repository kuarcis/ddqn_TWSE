import pandas_datareader.data as web_data
from pandas import ExcelWriter
import datetime, time
import os

vaild_tar = False
stock_target = '2330.TW'
while not vaild_tar:
    try:
        input_tar_tmp = input('enter yahoo finance data target (in the form of index:"^XXXX", stock:"XXXX.YY") (default:[%s]) ' %(stock_target))
        if input_tar_tmp.find('.') is False or input_tar_tmp[0] !='^' :
            raise NameError
        
        stock_target = input_tar_tmp
        print('set %s as target...' %(stock_target))
        vaild_tar = True
    except NameError:
        print('wrong data format, try again')
    



data_dist= '../data/'

def_end_date = datetime.date.today()-datetime.timedelta(days=1)
year_multipler = 5
def_date_range = datetime.timedelta(days=365)* year_multipler
def_start_date = def_end_date - def_date_range

#===============define date range selection function


def date_input_check(mode, daterange = year_multipler,start= def_start_date, end = def_end_date):
    date_check = False
    while date_check is not True:
        if mode == 1 :
            try:
                input_start = input('enter start date of data in YYYY-MM-DD format, default:[%s]' %(start))
                if input_start is '':
                    input_start = start
                else:
                    input_start = time.strptime(input_start,"%Y-%m-%d")

                input_end = input('enter end date of data in YYYY-MM-DD format, default:[%s]' %(end))
                if input_end is '':
                    input_end = end
                else:
                    input_end = time.strptime(input_end, "%Y-%m-%d")
                if vaild_date_check(input_start,input_end) is False:
                    raise NameError

                date_check = True

            except ValueError:
                print('wrong date format, please try again')
            except NameError:
                print('start date is late than end date, please enter again')
            except Exception as err:
                print(err)
        if mode == 2:
            try:
                input_end = input('enter end date of data in YYYY-MM-DD format, default:[%s]' %(end))
                if input_end is '':
                    input_end = end
                else:
                    input_end = time.strptime(input_end, "%Y-%m-%d")
                
                date_range = year_multipler
                input_date_range = input('enter trace back data range (in year), default:[%s]' %(date_range))
                if input_date_range is '':
                    input_date_range = date_range
                if int(input_date_range) >0:
                    input_date_range  = int(input_date_range)
                else:
                    print('wrong range data type, please try again')
                    raise Exception

                input_start = input_end - datetime.timedelta(days=365)* input_date_range
                
                date_check = True

            except ValueError:
                print('wrong date format, please try again')
            except Exception as err:
                print(err)

    return input_start, input_end

def vaild_date_check(start, end):
    if start >= end:
        return False
    return True



#================select date range mode===============
try:
    print('yahoo finance data grabber:')
    print('mode select')
    inputmode = False
    while not inputmode:
        try:    
            inputmode = input(' \
            \n1: grab data by select start date and end date  \
            \n2: grab data by select a end date and data range (in X years)  \
            \ndefault is 1, please enter: [1-2] ')
            
            if inputmode is '' or int(inputmode) == 1:
                inputmode = 1
                inputmode = int(inputmode)        
                print('select 1: grab data by select start date and end date')
            elif int(inputmode) == 2:
                inputmode = int(inputmode)
                print('select 2: grab data by select a end date and data range (in X years)')
            else:
                raise Exception
            
        except:
            inputmode = False
            print('wrong type of input, try again')

    input_start, input_end = date_input_check(mode = inputmode)

    data_filename = stock_target + '_'+ input_start.strftime("%Y%m%d") + '_' + input_end.strftime("%Y%m%d") +'.xls'
    data_loc = data_dist+data_filename



    data_src = [(stock_target, 'yahoo', input_start, input_end), (stock_target, 'yahoo-actions', input_start, input_end)]
    try:
        print('downloading history price ....')
        stock_hp = web_data.DataReader(*data_src[0])

        print('downloading corporate actions .....')
        try:
            stock_ca = web_data.DataReader(*data_src[1])
        except ValueError:
            print('no CA in target range, pass')

        print('all download done')


    except Exception as e:
        print('error occur:')
        print(e)

    if os.path.exists(data_loc):
        print('file with the same data range already exist, delete old file...')
        os.remove(data_loc)

    try:
        
        print('writing to '+data_loc+'....')
        d_writer = ExcelWriter(data_loc)
        stock_hp.to_excel(d_writer,'hp')
        print('history price data done')
        try:
            stock_ca.to_excel(d_writer,'ca')
            print('corporate action write done')
        except:
            print('no CA data exist, pass CA writing')
        d_writer.save()
        print('all write done')

    except Exception as e:
        print('error occur:')
        print(e)

except KeyboardInterrupt:
    print('keyboard interrupt, stop')