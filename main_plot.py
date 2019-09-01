from plot import * 
###From Pickle
modelType = 'alexnetv1-1-'
path = '/media/david/datos/Violence DATA/violentflows/Results'

train_lost = loadList(path+modelType+'train_lost.txt')
train_acc = loadList(path+modelType+'train_acc.txt')
test_lost = loadList(path+modelType+'test_lost.txt')
test_acc = loadList(path+modelType+'test_acc.txt')

num_epochs = int(len(train_lost)/5)
print('len: ',len(train_lost))
print('num_epochs size: ',num_epochs)

plotScalarFolds(train_acc,train_lost,num_epochs,'Train')
plotScalarFolds(test_acc,test_lost,num_epochs,'Test')