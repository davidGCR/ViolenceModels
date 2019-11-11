import sys
sys.path.insert(1, '/media/david/datos/PAPERS-SOURCE_CODE/MyCode')
import argparse
import ANOMALYCRIME.transforms_anomaly as transforms_anomaly
import torch
import ANOMALYCRIME.anomaly_initializeDataset as anomaly_initializeDataset
import saliencyTester
import constants
import os
import tkinter
from PIL import Image, ImageFont, ImageDraw, ImageTk
import numpy as np
import cv2
import glob

def tensor2numpy(x):
    x = x / 2 + 0.5
    x = x.numpy()
    x = np.transpose(x, (1, 2, 0))
    print('x: ', type(x), x.shape)
    return x

def get_anomalous_video(video_test_name, reduced = True):
    """ get anomalous video """
    label = video_test_name[:-3]
    path = os.path.join(constants.PATH_UCFCRIME2LOCAL_FRAMES_REDUCED, video_test_name) if reduced else os.path.join(constants.PATH_UCFCRIME2LOCAL_FRAMES, video_test_name)
    
    list_frames = os.listdir(path) 
    list_frames.sort()
    num_frames = len(glob.glob1(path, "*.jpg"))
    bdx_file_path = os.path.join(constants.PATH_UCFCRIME2LOCAL_BBOX_ANNOTATIONS, video_test_name+'.txt')
    data = []
    with open(bdx_file_path, 'r') as file:
        for row in file:
            data.append(row.split())
    data = np.array(data)
    bbox_infos_frames = []
    for frame in list_frames:
        num_frame = int(frame[len(frame) - 7:-4])
        if num_frame != int(data[num_frame, 5]):
            sys.exit('Houston we have a problem: index frame does not equal to the bbox file!!!')
            # print('Houston we have a problem: index frame does not equal to the bbox file!!!')
        flac = int(data[num_frame,6]) # 1 if is occluded: no plot the bbox
        xmin = int(data[num_frame, 1])
        ymin= int(data[num_frame, 2])
        xmax = int(data[num_frame, 3])
        ymax = int(data[num_frame, 4])
        info_frame = [frame, flac, xmin, ymin, xmax, ymax]
        bbox_infos_frames.append(info_frame)
    
    return path, label, bbox_infos_frames, num_frames
 
def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saliencyModelFile", type=str)
    parser.add_argument("--batchSize", type=int, default=3)
    # parser.add_argument("--numEpochs", type=int, default=10)
    parser.add_argument("--numWorkers", type=int, default=1)
    parser.add_argument("--numDiPerVideos", type=int, default=5)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--shuffle", type=lambda x: (str(x).lower() == 'true'), default=False)
    args = parser.parse_args()
    maxNumFramesOnVideo = 0
    videoSegmentLength = 16
    numDiPerVideos = args.numDiPerVideos
    positionSegment = 'random'
    num_classes = 2
    input_size = 224
    transforms = transforms_anomaly.createTransforms(input_size)
    dataset_source = 'frames'
    batch_size = args.batchSize
    num_workers = args.numWorkers
    saliency_model_file = args.saliencyModelFile
    shuffle = args.shuffle
    threshold = args.threshold
    print('shuffle: ', shuffle)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    saliency_model_file = args.saliencyModelFile
    saliency_model_config = saliency_model_file
    path_dataset = constants.PATH_UCFCRIME2LOCAL_FRAMES_REDUCED
    train_videos_path = os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'Train_split_AD.txt')
    test_videos_path = os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'Test_split_AD.txt')
    dataloaders_dict, test_names = anomaly_initializeDataset.initialize_final_only_test_anomaly_dataset(path_dataset, train_videos_path, test_videos_path, dataset_source, batch_size,
                                                        num_workers, numDiPerVideos, transforms, maxNumFramesOnVideo, videoSegmentLength, positionSegment, shuffle)
    tester = saliencyTester.SaliencyTester(saliency_model_file, num_classes,  dataloaders_dict['test'], test_names, input_size, saliency_model_config, numDiPerVideos, threshold)
    
    path, label, bbox_infos_frames, num_frames = get_anomalous_video('Burglary048', reduced=True)
    print(path, label, num_frames)
    for i, data in enumerate( dataloaders_dict['test'], 0):
        di_images, masks = tester.compute_sal_map(data)

        print('di_images: ', type(di_images), di_images.size())
        print('masks: ', type(masks), masks.size())
        di_images = torch.squeeze(di_images, 0).cpu()
        threshold_mask = masks.clone()
        
        masks = torch.squeeze(masks,0)
        masks = masks.detach().cpu()
        masks = masks.repeat(3, 1, 1)

        di_images = tensor2numpy(di_images)
        masks = tensor2numpy(masks)

        threshold_mask = torch.squeeze(threshold_mask,0)
        threshold_mask = threshold_mask.detach().cpu()
        threshold_mask = tester.normalize_tensor(threshold_mask)
        threshold_mask = tensor2numpy(threshold_mask)
        threshold_mask = tester.thresholding_cv2(threshold_mask)
        print('threshold_mask: ', type(threshold_mask), threshold_mask.shape)

        # threshold_mask = threshold_mask.repeat(3, 1, 1) 
        
        img_concate_Hori=np.concatenate((di_images,masks, threshold_mask),axis=1)
        cv2.imshow('eefef', img_concate_Hori)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break
       
    # saliencyTester.test(saliency_model_file, num_classes, dataloaders_dict['test'], test_names, input_size, saliency_model_config, numDiPerVideos, threshold)

__main__()

