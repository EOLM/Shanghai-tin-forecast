import pandas
import numpy as np
def getdata():
    data=pandas.read_csv("./data.csv",dtype = {'日期2' : str,"收盘":float,"开盘":float,'高':float,'低':float,'交易量':str,'涨跌幅':str,'日期':int},encoding='gbk')

    dataset=data['收盘'].values
    dataset = dataset.astype('float32')
    train_size = int(len(dataset) * 0.8)
    trainlist = dataset[:train_size]
    testlist = dataset[train_size:]
    return trainlist,testlist

def create_data(dataset,look_back):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX),np.array(dataY)