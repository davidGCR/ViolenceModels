import numpy as np
import cv2
import time

# video_path = '/media/david/datos/Violence DATA/HockeyFights/fi1_xvid.avi'
# video_path = 'fight_0001.mpeg'
video_path = 'fi1_xvid.avi'
cap = cv2.VideoCapture(video_path)

capture_duration = 1

start_time = time.time()
images = []

# while(int(time.time() - start_time)<capture_duration):
while(True):
    ret, frame = cap.read()
    print(ret)
    # img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow('frame',frame)
    images.append(frame)

print('number of frames: ',len(images))
