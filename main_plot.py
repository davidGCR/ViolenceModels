from plot import *
import matplotlib.pyplot as plt
import argparse

###From Pickle
# modelType = 'alexnetv2-frames-Finetuned:False-2-decay-'
# path = '/media/david/datos/Violence DATA/violentflows/Results/frames/'
# modelType = 'alexnetv2-frames-Finetuned:False-5-decay-'
# path = '/media/david/datos/Violence DATA/HockeyFights/Results/frames/'

def plot_results(path, modelType,lastEpoch):
    train_lost = loadList(path+modelType+'train_lost.txt')
    train_acc = loadList(path+modelType+'train_acc.txt')
    test_lost = loadList(path+modelType+'test_lost.txt')
    test_acc = loadList(path+modelType+'test_acc.txt')

    num_epochs = int(len(train_lost)/5)
    print('len: ',len(train_lost))
    print('num_epochs size: ', num_epochs)

    fig2 = plt.figure(figsize=(12,12))

    plotScalarFolds(train_acc,train_lost,num_epochs,'Train',fig2,3,2,1)
    plotScalarFolds(test_acc, test_lost, num_epochs, 'Test',fig2,3,2,3)

    avgTrainAcc = getAverageFromFolds(train_acc,num_epochs)
    avgTrainLost = getAverageFromFolds(train_lost,num_epochs)
    avgTestAcc = getAverageFromFolds(test_acc,num_epochs)
    avgTestLost = getAverageFromFolds(test_lost, num_epochs)

    plotScalarCombined(avgTrainLost,avgTestLost, num_epochs,'Error Promedio','Error',fig2,3,2,5)
    plotScalarCombined(avgTrainAcc, avgTestAcc, num_epochs, 'Tasa de Acierto Promedio', 'Tasa de Acierto',fig2,3,2,6)

    plt.show()

    # max_acc_train = []
    # max_acc_test = []
    # lastEpoch =num_epochs

    print('max test accuracy until ',lastEpoch,' epoch: ', np.amax(np.array(avgTestAcc[0:lastEpoch])))
    # avgTestAcc[0:lastEpoch]

def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resultsDir', type=str, default='/media/david/datos/Violence DATA/HockeyFights/Results/frames/', help='Directory containing results')
    parser.add_argument('--modelType', type=str, help='model name')
    parser.add_argument('--lastEpoch', type=int, default=100, help='last epoch before overfiting')
    args = parser.parse_args()
    path = args.resultsDir
    modelType = args.modelType
    lastEpoch = args.lastEpoch

    plot_results(path, modelType, lastEpoch)

__main__()