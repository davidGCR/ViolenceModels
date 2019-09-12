from dinamycImage import *
import cv2
import argparse

def __main__():
  parser = argparse.ArgumentParser()
  parser.add_argument("--video_path", type=str, default="/media/david/datos/Violence DATA/HockeyFights/videos/violence/fi1_xvid.avi", help="Directory containing video")
  parser.add_argument("--seq_duration", type=float, default=1, help="time to compute dynamic image")
  args = parser.parse_args()
  vid_name = args.video_path
  seq_duration = args.seq_duration
  cap = cv2.VideoCapture(vid_name)
  video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  fps = cap.get(cv2.CAP_PROP_FPS)
  duration = video_length / fps
  print('fps:',fps,'video duration:',duration)
  count = 0
  frames = []
  current_time = 0
  while count < video_length - 1:
  # while current_time < duration:
    current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
    print('current_time ',current_time, 'seq_duration ',seq_duration)
    
    if current_time > seq_duration:
      print('========================\t num frames:', len(frames))
      diPIL, di = getDynamicImage(frames)
      cv2.imshow('Di', di)
      frames = []
      seq_duration = current_time + seq_duration
      
    success, image = cap.read()
    if success:
        frames.append(image)
        cv2.imshow('Frame', image)
        if cv2.waitKey(70) & 0xFF == ord('q'):
          break
    count += 1
  
  if len(frames) > 0:
    diPIL, di = getDynamicImage(frames)
    cv2.imshow('Di', di)
    cv2.waitKey(70)

__main__()