import cv2
import numpy as np
import darknet
import os
import time

import mask_parameter as mask_parameter
from threading import Thread, Lock
from multiprocessing import Process, Queue


from datetime import datetime
import pandas as pa


write_path = 'add write path here'

cameraSource = "source address of the camera"

#Use the trained CNN model here
configPath = "Config file path"
weightPath = "weight file path"
metaPath = "metadata path"



def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cordConverter(detections, class_list):
    detBox = []
    for detection in detections:
        class_ = class_list.index(detection[0].decode('ASCII'))
        score = detection[1]
        x, y, w, h = detection[2][0], \
                     detection[2][1], \
                     detection[2][2], \
                     detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))

        detBox.append([xmin, ymin, xmax, ymax, score, class_])
    detBox = np.array(detBox)
    return detBox


class CameraStream(object):
    def __init__(self, src):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = Lock()

    def start(self):
        if self.started:
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.started:
            (grabbed, frame) = self.stream.read()
            self.read_lock.acquire()
            self.grabbed, self.frame = grabbed, frame
            self.read_lock.release()

    def read(self):
        self.read_lock.acquire()
        frame = self.frame.copy()
        self.read_lock.release()
        return frame

    def stop(self):
        self.started = False
        self.thread.join()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stream.release()


def mask_creater(shape, roi_indicater):
    mask = np.zeros(shape, dtype=np.uint8)
    if roi_indicater == 1:
        roi_corners = mask_parameter.roi_corners_1
    elif roi_indicater == 2:
        roi_corners = mask_parameter.roi_corners_2
    elif roi_indicater == 3:
        roi_corners = mask_parameter.roi_corners_3
    elif roi_indicater == 4:
        roi_corners = mask_parameter.roi_corners_4

    cv2.fillPoly(mask, roi_corners, (1, 1, 1))
    return mask


def img_operations(frame, mask, split):
    if frame.shape != (576, 704, 3):
        frame = cv2.resize(frame, (704, 576))

    frame = frame * mask
    frame = frame[split[1]:, split[0]:]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (416, 416), interpolation=cv2.INTER_LINEAR)
    return frame


def cvDrawBoxes(detections, img):
    detections = detections.tolist()
    for detection in detections:
        xmin = int(detection[0])
        ymin = int(detection[1])
        xmax = int(detection[2])
        ymax = int(detection[3])

        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img,
                    str(int(detection[4])),
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    [0, 0, 0], 1)
    return img



# load once
netMain = None
metaMain = None
altNames = None
if netMain is None:
    netMain = darknet.load_net_custom(configPath.encode(
        "ascii"), weightPath.encode("ascii"), 0, 1) 
if metaMain is None:
    metaMain = darknet.load_meta(metaPath.encode("ascii"))
if altNames is None:
    try:
        with open(metaPath) as metaFH:
            metaContents = metaFH.read()
            import re

            match = re.search("names *= *(.*)$", metaContents,
                              re.IGNORECASE | re.MULTILINE)
            if match:
                result = match.group(1)
            else:
                result = None
            try:
                if os.path.exists(result):
                    with open(result) as namesFH:
                        namesList = namesFH.read().strip().split("\n")
                        altNames = [x.strip() for x in namesList]
            except TypeError:
                pass
    except Exception:
        pass
darknet_image = darknet.make_image(darknet.network_width(netMain),
                                   darknet.network_height(netMain), 3)


def read_images(queue_channel1_):
    shape_img = (576, 704, 3)
    mask_1 = mask_creater(shape_img, 1)


    split_1 = mask_parameter.split_1


    camRead = CameraStream(src=cameraSource).start()

    prev_time = 0

    flag = False
    temp_stop = False
    while True:
        
            if flag & (not temp_stop):
                if t1 - prev_time > 1 / 15:

                    splittedFrames = img_operations(camRead.read(), mask_1, split_1)

                    queue_channel1_.put(frame1_)
 

                    if ((t1 - prev_time) > 1 / 10) & ((t1 - prev_time) < 1 / 8):
                        pass


def visualize(queue_track1, queue_track2):
    while True:
        if (queue_track1.qsize() != 0) & (queue_track2.qsize() != 0):
            track_info1 = queue_track1.get()
            track_info2 = queue_track2.get()

            if (type(track_info1) == str) & (type(track_info2) == str):

                cv2.destroyAllWindows()
                if track_info1 == 'end':
                    break
                continue
            else:
                img1 = cv2.cvtColor(track_info1[0], cv2.COLOR_RGB2BGR)
                img2 = cv2.cvtColor(track_info2[0], cv2.COLOR_RGB2BGR)
                img1 = cvDrawBoxes(track_info1[1], img1)
                img2 = cvDrawBoxes(track_info2[1], img2)

                cv2.imshow('channel1', img1)
                cv2.imshow('channel2', img2)
                cv2.waitKey(int(1000 / 15))
        else:
            continue


if __name__ == '__main__':
    
    frameQueue= Queue()

    detectionQueue = Queue()

    node = Process(target=read_images, args=(frameQueue))
    node.start()
    node.join()
