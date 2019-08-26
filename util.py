import pickle

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False



def saveList(name, lt):
  data_file = '/content/drive/My Drive/VIOLENCE DATASETS/HockeyFightsFrames/AlexNetV1-4di'+str(name)+'.txt'
  with open(data_file, 'wb') as filehandle:
      # store the data as binary data stream
      pickle.dump(lt, filehandle)
def loadList(name):
  with open(name, 'rb') as filehandle:
    # read the data as binary data stream
    hist2 = pickle.load(filehandle)
    return hist2