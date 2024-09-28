from cv2 import bitwise_and
from sympy.physics.units import length
from torch import dtype
from torch.onnx.symbolic_opset11 import vstack
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import numpy as np


cap=cv2.VideoCapture("../Yolo With Video/cars.mp4")
mask=cv2.imread("mask.png")
model = YOLO("yolov8n.pt")
# Tracking
tracker = Sort(max_age=20,min_hits=3,iou_threshold=0.3)
limit = (50,420,673,420)

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

Count = 0
while True:
    success, img=cap.read()
    imgRegion = bitwise_and(img,mask)
    results = model(imgRegion,stream=True)
    detections = np.empty((0, 5))

    line=cv2.line(img,(limit[0],limit[1]),(limit[2],limit[3]),(0,0,255),5)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
            w,h = x2 - x1, y2-y1
            # cvzone.cornerRect(img,(x1,y1,w,h),l=8)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255))
            conf = math.ceil((box.conf[0])*100)/100
            cls = int(box.cls[0])
            if conf > 0.3 :
                # cvzone.putTextRect(img,f'{classNames[cls]} {conf}',(max(0,x1),max(0,y1)),scale=1,thickness=1,offset=8)
                current = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, current))
    resultsTracker = tracker.update(detections)
    for result in resultsTracker:
        x1,y1,x2,y2,id = result
        x1,y1,x2,y2,id = int(x1), int(y1), int(x2), int(y2), int(id)
        print( x1,y1,x2,y2,id)
        if classNames[cls] =="car" and conf > 0.3 :
            cvzone.cornerRect(img,(x1,y1,w,h),l=8)
            cvzone.putTextRect(img,f'{id}',(max(0,x1),max(0,y1)),scale=2,thickness=2,offset=15)
            cx,cy = x1+w//2 , y1+h//2
            cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)
        if limit[0]<cx<limit[2] and limit[1]-20<cy<limit[1]+20:
            Count+=1

    cvzone.putTextRect(img,f'TotalCount={Count}',(20,50),colorR=(255,0,255),colorT=(0,0,0))

    cv2.imshow("Image",img)
    cv2.imshow("ImageRegion",imgRegion)
    cv2.waitKey(0)
