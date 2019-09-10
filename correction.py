import pandas as pd
from util import *

train_file = 'train.txt'
test_file = 'test.txt'

df = pd.read_csv(train_file)
trainloss = df['loss'].tolist()
trainacc = df['acc']

dft = pd.read_csv(test_file)
testloss = dft['loss'].tolist()
testacc = dft['acc'].tolist()
path = '/media/david/datos/Violence DATA/HockeyFights/Results/frames/'
modelType = 'alexnet-frames-Finetuned:False-4di-tempMaxPool-OnPlateau-'
saveList2(path + modelType + 'train_lost.txt', trainloss)
saveList2(path + modelType + 'train_acc.txt', trainacc)
saveList2(path + modelType + 'test_lost.txt', testloss)
saveList2(path+modelType+'test_acc.txt',testacc)
print(testacc)