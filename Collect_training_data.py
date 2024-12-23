#================================================================

#   Description : YOLO detection to XML example script

#================================================================
import os
import subprocess
import time
from datetime import datetime
import cv2
import mss
import numpy as np
import tensorflow as tf
from yolov3.utils import *
from yolov3.configs import *
from yolov3.yolov4 import read_class_names
from tools.Detection_to_XML import CreateXMLfile
import random

import pyttsx3
#from .engine import Engine
engine = pyttsx3.init()

def draw_enemy(image, bboxes, CLASSES=YOLO_COCO_CLASSES, show_label=True, show_confidence = True, Text_colors=(255,255,0), rectangle_colors='', tracking=False):
    NUM_CLASS = read_class_names(CLASSES)
    num_classes = len(NUM_CLASS)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    detection_list = []

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = rectangle_colors if rectangle_colors != '' else colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 1000)
        if bbox_thick < 1: bbox_thick = 1
        fontScale = 0.75 * bbox_thick
        (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])

        # put object rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, bbox_thick*2)

        x, y = int(x1+(x2-x1)/2), int(y1+(y2-y1)/2)

        if show_label:
            # get text label
            score_str = " {:.2f}".format(score) if show_confidence else ""

            if tracking: score_str = " "+str(score)

            label = "{}".format(NUM_CLASS[class_ind]) + score_str

            # set value of image label/object name
            print('POSSIBLE SCENES ARE: ' + label)
            engine.say("Warning")
            engine.say(label)
            # get text size
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                                  fontScale, thickness=bbox_thick)
            # put filled text rectangle
            cv2.rectangle(image, (x1, y1), (x1 + text_width, y1 - text_height - baseline), bbox_color, thickness=cv2.FILLED)

            # put text above rectangle
            cv2.putText(image, label, (x1, y1-4), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, Text_colors, bbox_thick, lineType=cv2.LINE_AA)

    return image

def detect_enemy(Yolo, original_image, input_size=416, CLASSES=YOLO_COCO_CLASSES, score_threshold=0.3, iou_threshold=0.45, rectangle_colors=''):
    image_data = image_preprocess(original_image, [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    if YOLO_FRAMEWORK == "tf":
        pred_bbox = Yolo.predict(image_data)

    elif YOLO_FRAMEWORK == "trt":
        batched_input = tf.constant(image_data)
        result = Yolo(batched_input)
        pred_bbox = []
        for key, value in result.items():
            value = value.numpy()
            pred_bbox.append(value)

    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)

    bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)
    bboxes = nms(bboxes, iou_threshold, method='nms')

    image = draw_enemy(original_image, bboxes, CLASSES=CLASSES, rectangle_colors=rectangle_colors)


    #   Voice output for object detection
    #                 car                    bus                  truck
        if CLASSES[0][i] == 3 or CLASSES[0][i] == 6 or CLASSES[0][i] == 8:
          if score[0][i] >= 0.5:
            mid_x = (bboxes[0][i][1]+bboxes[0][i][3])/2
            mid_y = (bboxes[0][i][0]+bboxes[0][i][2])/2
            apx_distance = round(((1 - (bboxes[0][i][3] - bboxes[0][i][1]))**4),1)
            cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            if apx_distance <=0.5:
              if mid_x > 0.3 and mid_x < 0.7:
                cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                print("Warning" + label + "Ahead")
                engine.say("Warning" + label + "Ahead")
                engine.runAndWait()


    return image, bboxes

offset = 30
times = []
sct = mss.mss()
yolo = Load_Yolo_model()
while True:
    t1 = time.time()
    img = np.array(sct.grab({"top": 87-offset, "left": 1920, "width": 1280, "height": 720, "mon": -1}))
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    image, bboxes = detect_enemy(yolo, np.copy(img), input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
    if len(bboxes) > 0:
        CreateXMLfile("XML_Detections", str(int(time.time())), img, bboxes, read_class_names(TRAIN_CLASSES))
        print("got it")
        time.sleep(2)

    t2 = time.time()
    times.append(t2-t1)
    times = times[-20:]
    ms = sum(times)/len(times)*1000
    fps = 1000 / ms
    print("FPS", fps)
