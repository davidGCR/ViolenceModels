import sys
sys.path.insert(1, '/media/david/datos/PAPERS-SOURCE_CODE/MyCode')
import os
import constants
import glob
import numpy as np
import cv2
import math
from point import Point
from bounding_box import BoundingBox
# import point.Point as Point



def join2NearBBoxes(bbox1, bbox2, max_distance):
    # xmin_1 = bbox1[0]
    # ymin_1 = bbox1[1]
    # xmax_1 = bbox1[2]
    # ymax_1 = bbox1[3]

    # xmin_2 = bbox2[0]
    # ymin_2 = bbox2[1]
    # xmax_2 = bbox2[2]
    # ymax_2 = bbox2[3]

    # xc_1, yc_1 = center(xmin_1, ymin_1, xmax_1, ymax_1)
    # xc_2, yc_2 = center(xmin_2, ymin_2, xmax_2, ymax_2)
    return 0

# def isInside(center, box):
#     return center[]<


def distance(p1, p2):
    distance = math.sqrt(((p1.x - p2.x)** 2) + ((p1.y - p2.y)** 2))
    return distance
# def distance(p1, p2):
#     distance = math.sqrt(((p1[0] - p2[0])** 2) + ((p1[1] - p2[1])** 2))
#     return distance

# def center(xmin, ymin, xmax, ymax):
#     xcent_1 = xmin + int((xmax - xmin) / 2)
#     ycent_1 = ymin + int((ymax - ymin) / 2)
#     return xcent_1, ycent_1

def computeBoundingBoxFromMask(mask):
    """
    *** mask: rgb numpy image
    """

    mask = thresholding_cv2(mask)
    # mask = mask.astype('uint8')
    # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # print('ccccc: ', mask.shape)
    img = process_mask(mask)
    img, contours = findContours(img, remove_fathers=True)
    
    bboxes = bboxes_from_contours(img, contours)
    print(len(bboxes),bboxes)
    return bboxes

def process_mask(img):
    kernel_exp = np.ones((5, 5), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_exp)
    kernel_dil = np.ones((7, 7), np.uint8)
    img = cv2.dilate(img, kernel_dil, iterations=1)
    kernel_clo = np.ones((11, 11), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_clo)
    return img

def findContours(img, remove_fathers = True):
    # Detect edges using Canny
    # canny_output = cv2.Canny(src_gray, threshold, threshold * 2)
    # Find contours
    # color = cv2.Scalar(0, 255, 0)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours
    drawing = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    if remove_fathers:
        removed = []
        for idx,contour in enumerate(contours):
            if hierarchy[0, idx, 3] == -1:
                removed.append(contour)
        contours = removed
    # print('contours: ', len(contours))
    for i in range(len(contours)):
        cv2.drawContours(drawing, contours, i, (0, 255, 0), 2, cv2.LINE_8, hierarchy, 0)
    # Show in a window
    # cv2.imshow('Contours', drawing)
    return drawing, contours

def plotBBoxesOnImage(img, bboxes, color=(0, 0, 255)):
    shape = img.shape
    
    if shape[2] == 1:
        img = np.squeeze(img,2)
        img = gray2rgbRepeat(img)
    # print('ttttttttttt', img.shape)
    for i, box in enumerate(bboxes):
        cv2.rectangle(img,(box.pmin.x,box.pmin.y),(box.pmax.x,box.pmax.y),color,2)
        # cv2.rectangle(img, (int(bboxes[i][0]),int(bboxes[i][1])),
        #                 (int(bboxes[i][0] + bboxes[i][2]),int(bboxes[i][1] + bboxes[i][3])),
        #                 color, 2)
    return img

def bboxes_from_contours(img, contours):
    contours_poly = [None]*len(contours)
    boundRect = [None] * len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
    # drawing = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    bboxes = []
    for i, rect in enumerate(boundRect):
        print('REct: ', rect)
        bb = cvRect2BoundingBox(rect)
        bboxes.append(bb)
    #     color_red = (0,0,255)
    #     # cv2.drawContours(drawing, contours_poly, i, color)
    #     cv2.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color_red, 2)
    return bboxes

def cvRect2BoundingBox(cvRect):
    pmin = Point(cvRect[0], cvRect[1])
    pmax = Point(cvRect[0]+cvRect[2], cvRect[1]+cvRect[3])
    bb = BoundingBox(pmin,pmax)
    # print('bbbbbbbbxxxxx: ', pmin.x, bb.center.x)
    return bb
    


def thresholding_cv2(x):
        x = 255*x #between 0-255
        x = x.astype('uint8')
        # th = cv2.adaptiveThreshold(x,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,2)
        # Otsu's thresholding
        x = cv2.GaussianBlur(x,(5,5),0)
        # print('x numpy: ', x.shape, x.dtype)
        ret2, th = cv2.threshold(x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return th

def normalize_tensor(self, img):
        # print("normalize:", img.size())
        _min = torch.min(img)
        _max = torch.max(img)
        # print("min:", _min.item(), ", max:", _max.item())
        return (img - _min) / (_max - _min)

def get_anomalous_video(video_test_name, reduced_dataset = True):
    """ get anomalous video """
    label = video_test_name[:-3]
    path = os.path.join(constants.PATH_UCFCRIME2LOCAL_FRAMES_REDUCED, video_test_name) if reduced_dataset else os.path.join(constants.PATH_UCFCRIME2LOCAL_FRAMES, video_test_name)
    
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

def tensor2numpy(x):
    x = x / 2 + 0.5
    x = x.numpy()
    x = np.transpose(x, (1, 2, 0))
    # print('x: ', type(x), x.shape)
    return x

def rgb2grayUnrepeat(x):
    x = x[:,:, 0]
    return x

def gray2rgbRepeat(x):
    x = np.stack([x, x, x], axis=2)
    return x