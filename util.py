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



def saveList(path_out,model,curve, numDI, lt):
  data_file = path_out+'/'+str(model)+'-'+str(numDI)+'-'+str(curve)+'.txt'
  with open(data_file, 'wb') as filehandle:
      # store the data as binary data stream
      pickle.dump(lt, filehandle)
      print('saved ... ',data_file)


def loadList(name):
  with open(name, 'rb') as filehandle:
    # read the data as binary data stream
    hist2 = pickle.load(filehandle)
    return hist2

#######################################################################################
################################# Videos to Frames ####################################
#######################################################################################

def video2Images2(video_path, path_out):
  vid = cv2.VideoCapture(video_path)
  index_frame = 1
  while(True):
      ret, frame = vid.read()
      if not ret: 
          break
      name = path_out+'/'+'frame' + str("{0:03}".format(index_frame)) + '.jpg'
      print ('Creating...' + name)
      cv2.imwrite(name, frame)
      index_frame += 1 

def videos2ImagesFromKfols(path_videos, path_frames):
  list_folds = os.listdir(path_videos) ## [1 2 3 4 5]
  for fold in list_folds:
    violence_videos_path = path_videos+'/'+fold+'/Violence'
    nonviolence_videos_path = path_videos+'/'+fold+'/NonViolence'
    
    violence_videos_path_out = path_frames+'/'+fold+'/Violence'
    nonviolence_videos_path_out = path_frames+'/'+fold+'/NonViolence'
    
    violent_videos_paths = os.listdir(violence_videos_path)
    nonviolent_videos_paths = os.listdir(nonviolence_videos_path)
    
    for video in violent_videos_paths:
      frames_path_out = os.path.join(path_frames,fold,'Violence',os.path.splitext(video)[0])
      print(frames_path_out)
      if not os.path.exists(frames_path_out):
        os.makedirs(frames_path_out)
      video2Images2(os.path.join(violence_videos_path,video), frames_path_out)
      
    for video in nonviolent_videos_paths:
      frames_path_out = os.path.join(path_frames,fold,'NonViolence',os.path.splitext(video)[0])
      print(frames_path_out)
      if not os.path.exists(frames_path_out):
        os.makedirs(frames_path_out)
        
      video2Images2(os.path.join(nonviolence_videos_path,video), frames_path_out)
