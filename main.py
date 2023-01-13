import argparse
from dataprocess import getdata,create_data
from model import get_model
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100, help='epochs')
parser.add_argument('--look_back', type=int, default=8, help='look_back')
parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
args = parser.parse_args()

train_list,test_list=getdata()
trainX,trainY  = create_data(train_list,args.look_back)

testX,testY = create_data(test_list,args.look_back)

trainX=np.expand_dims(trainX,axis=2)
testX=np.expand_dims(testX,axis=2)

trainY=np.expand_dims(trainY,axis=1)
testY=np.expand_dims(testY,axis=1)

# 数据数值比较大 进行一定程度的缩放
trainX=trainX/1000
testX=testX/1000
trainY=trainY/1000
testY=testY/1000

model=get_model()

model.fit(trainX, trainY, epochs=args.epochs, batch_size=args.batch_size, verbose=1)

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
plt.title("Performance on train dataset")
plt.plot(trainY,c='b',)
plt.plot(trainPredict[1:],c='r')
plt.show()

plt.title("Performance on test dataset")
plt.plot(testY,c='b',)
plt.plot(testPredict[1:],c='r')
plt.show()
