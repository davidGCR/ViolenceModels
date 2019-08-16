from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import cv2
import torch
import glob
import time
from dinamycImage import *

class ViolenceDatasetVideos(Dataset):
  
  def __init__(self, dataset, labels, spatial_transform, seqLen=0,ptime=0.0):
    """
    Args:
        dataset (list): Paths to the videos.
        labels (list): labels froma data
        seqLen (int): Number of frames in each segment
        
        spatial_transform (callable, optional): Optional transform to be applied
            on a sample.
    """
    self.spatial_transform = spatial_transform
    self.images = dataset
    self.labels = labels
    self.seqLen = seqLen
    self.ptime = ptime

  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):
    vid_name = self.images[idx]
    label = self.labels[idx]
    inpSeq = []

    if self.seqLen == 0 and self.ptime != 0.0:  ##From videos
      video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
      fps = cap.get(cv2.CAP_PROP_FPS)
      # duration = video_length / fps
      
      cap = cv2.VideoCapture(vid_name)
      start_time = time.time()
      frames = []
      success = True
      fcount = 0

      while(time.time() - start_time < self.ptime and ):
        success, frame = cap.read()
        if not success:
            break
        fcount = fcount + 1
        # print(fcount)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # plt.imshow(img)
        # plt.show()
        frames.append(Image.fromarray(frame))

    elif self.seqLen != 0 and self.ptime == 0.0: ##From frames folder
      frames_list = os.listdir(vid_name)
      frames_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
      total_frames = len(frames_list)
      
      sequences = [frames_list[x:x+self.seqLen] for x in range(0, total_frames, self.seqLen)]
      
      # inpSeq = []
      
      for index, seq in enumerate(sequences):
        if len(seq)==self.seqLen:
          frames = []
          for frame in seq:
            img_dir = str(vid_name)+'/'+ frame
            
            img = Image.open(img_dir).convert("RGB")
            img = np.array(img) 
            frames.append(img)
          img = getDynamicImage(frames)
          inpSeq.append(self.spatial_transform(img.convert('RGB')))
    
    inpSeq = torch.stack(inpSeq, 0)
   
    return inpSeq, label


def createDataset(path_violence,path_noviolence):
  imagesF = []

  list_violence = os.listdir(path_violence)
  list_violence.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

  for target in list_violence:
      d = os.path.join(path_violence, target)
  #     print(d)
  #     if not os.path.isdir(d):
  #         continue
      imagesF.append(d)


  imagesNoF = []
  list_no_violence = os.listdir(path_noviolence)
  list_no_violence.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

  for target in list_no_violence:
      d = os.path.join(path_noviolence, target)
#       print(d)
  #     if not os.path.isdir(d):
  #         continue
      imagesNoF.append(d)

  Dataset = imagesF + imagesNoF
  Labels = list([1] * len(imagesF)) + list([0] * len(imagesNoF))
  NumFrames = [len(glob.glob1(Dataset[i], "*.jpg")) for i in range(len(Dataset))]
  return Dataset, Labels, NumFrames