import numpy as np
import cv2
import time
from dinamycImage import *

video_path = '/content/drive/My Drive/VIOLENCE DATASETS/HockeyFightsVideos/Fights/fi2_xvid.avi'
cap = cv2.VideoCapture(video_path)
start_time_ms = time.time()
vidcap = cv2.VideoCapture(video_path)

video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
duration = video_length/fps
print('video duration: ',str(duration%60),',  total frames on video: ',video_length)


diference_max = 3
interval_duration = 0.5
count = 0
success = True
frames = []
dinamycImages = []

numberDi = int(duration/interval_duration)
numberFramesInterval = 0
print('Number of Dynamic images estimated: ',numberDi)

# while success and vidcap.get(cv2.CV_CAP_PROP_POS_MSEC)/1000 < start_time_ms:
#     success, image = vidcap.read()
capture_duration = interval_duration
while(count<video_length-1):
  current_time = vidcap.get(cv2.CAP_PROP_POS_MSEC)/1000
  if current_time <= capture_duration:
#     print('time cr ', current_time)
    success, image = vidcap.read()
    if success:
      frames.append(image)
#       print('number of frames: ',len(frames))
      print('Read a new frame (success, count, time): ', success, count,current_time,'/',capture_duration)
  else: 
    numberFramesInterval = len(frames)
    print('number of frames to sumarize: ',numberFramesInterval)
    dinamycImages.append(getDynamicImage(frames))
    print('------------------------------------------------')
    frames = []
    frames.append(image)
#     print('number of frames: ',len(frames))
    capture_duration = current_time + interval_duration
  count += 1
  
if(numberFramesInterval - len(frames) < diference_max):
  print('number of frames to sumarize: ',len(frames))
  dinamycImages.append(getDynamicImage(frames))
  
print('number of dynamic images: ',len(dinamycImages))