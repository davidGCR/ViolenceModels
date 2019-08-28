import pickle
import os
import glob
import cv2
import numpy as np


def createDataset(path_violence, path_noviolence):
    imagesF = []

    list_violence = os.listdir(path_violence)
    list_violence.sort(key=lambda f: int("".join(filter(str.isdigit, f))))

    for target in list_violence:
        d = os.path.join(path_violence, target)
        imagesF.append(d)
    imagesNoF = []
    list_no_violence = os.listdir(path_noviolence)
    list_no_violence.sort(key=lambda f: int("".join(filter(str.isdigit, f))))

    for target in list_no_violence:
        d = os.path.join(path_noviolence, target)
        imagesNoF.append(d)

    Dataset = imagesF + imagesNoF
    Labels = list([1] * len(imagesF)) + list([0] * len(imagesNoF))
    NumFrames = [len(glob.glob1(Dataset[i], "*.jpg")) for i in range(len(Dataset))]
    return Dataset, Labels, NumFrames

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False



def saveList(model,curve, numDI, lt):
  data_file = '/content/drive/My Drive/VIOLENCE DATASETS/HockeyFightsFrames/'+str(model)+'-'+str(numDI)+'-'+str(curve)+'.txt'
  with open(data_file, 'wb') as filehandle:
      # store the data as binary data stream
      pickle.dump(lt, filehandle)


def loadList(name):
  with open(name, 'rb') as filehandle:
    # read the data as binary data stream
    hist2 = pickle.load(filehandle)
    return hist2


