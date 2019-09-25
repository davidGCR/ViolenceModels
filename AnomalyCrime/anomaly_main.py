import sys
sys.path.insert(1, '/media/david/datos/PAPERS-SOURCE_CODE/MyCode')
from dataset import *
import os
import re
from util import video2Images2

def videos2frames(path_videos, path_frames):
  listViolence = os.listdir(path_videos)
#   listViolence.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
  print(listViolence)
#   for idx,video in enumerate(listViolence):
#     path_video = os.path.join(path_videos, 'violence', video)
#     path_frames_out = os.path.join(path_frames, 'violence', str(idx+1))
#     if not os.path.exists(path_frames_out):
#         os.makedirs(path_frames_out)
#     video2Images2(path_video, path_frames_out)

def createAnomalyDataset(path_violence,path_noviolence):
    imagesF = []
    list_violence = os.listdir(path_violence)
    list_violence.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    r = re.compile("([a-zA-Z]+)([0-9]+)")
    labels = [r.match(string).groups() for string in list_violence]
    labels = [tup[0] for tup in labels]
    classes = {'Normal':0, 'Arrest':1, 'Assault':2, 'Burglary':3, 'Robbery':4, 'Stealing':5, 'Vandalism':6}
    labels_int = [classes[label] for label in labels]

    print(classes)
    print()
    print(labels)
    print(labels_int)

    for target in list_violence:
        d = os.path.join(path_violence, target)
        imagesF.append(d)
    
    print(imagesF)
    # imagesNoF = []
    # list_no_violence = os.listdir(path_noviolence)
    # list_no_violence.sort(key=lambda f: int("".join(filter(str.isdigit, f))))

    # for target in list_no_violence:
    #     d = os.path.join(path_noviolence, target)
    #     imagesNoF.append(d)

    # Dataset = imagesF + imagesNoF
    # Labels = list([1] * len(imagesF)) + list([0] * len(imagesNoF))
    # NumFrames = [len(glob.glob1(Dataset[i], "*.jpg")) for i in range(len(Dataset))]
    # return Dataset, Labels, NumFrames

def __main__():
    # Dataset, Labels, NumFrames =
    # createAnomalyDataset('/media/david/datos/Violence DATA/AnomalyCRIME/UCFCrime2Local/videos', '')
    videos2frames('/media/david/datos/Violence DATA/AnomalyCRIME/UCFCrime2Local/videos', '/media/david/datos/Violence DATA/AnomalyCRIME/UCFCrime2Local/frames')

__main__()