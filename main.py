import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.models import resnext50_32x4d

from preprocessing import transform


def setup_resnet_50():
    resnet_50 = resnext50_32x4d(pretrained=False)
    Net = nn.Sequential()
    Net.add_module("resnet_50", resnet_50)
    Net.add_module("FC", nn.Linear(1000, 2))
    maskNet = nn.DataParallel(Net).to(device)
    maskNet.load_state_dict(
        torch.load(
            "model_resnet_50_stage2.pth",
            map_location=torch.device(device)
        )
    )
    return maskNet


os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'

device = "cuda:0" if torch.cuda.is_available() else "cpu"

face_cascade = cv2.dnn.readNet("face detector/deploy.prototxt",
                               "face detector/res10_300x300_ssd_iter_140000.caffemodel")
cap = cv2.VideoCapture(0)

maskNet = setup_resnet_50()
maskNet.eval()

while True:
    _, img = cap.read()
    img = cv2.flip(img, 1)
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img,
                                 1.0,
                                 (224, 224),
                                 (104.0, 177.0, 123.0),
                                 False,
                                 False)

    face_cascade.setInput(blob)
    detections = face_cascade.forward()
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face_img = img[startY:endY, startX:endX]
            face_img_RBG = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            PIL_image = Image.fromarray(face_img_RBG)
            inp = transform(PIL_image).unsqueeze(0)
            output = maskNet(inp)
            output = F.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)
            prob = output[0][pred]
            result, confidence = pred.item(), round(prob.item() * 100, 2)
            cv2.rectangle(img,
                          (startX, startY),
                          (endX, endY),
                          (0, 255, 0) if result == 1 else (0, 0, 255),
                          2)
            cv2.putText(img,
                        f"With Mask: {confidence}%" if result == 1 else f"Without Mask: {confidence}%",
                        (startX + 5, startY - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0) if result == 1 else (0, 0, 255),
                        1,
                        cv2.LINE_4)

    cv2.imshow('Video', img)

    k = cv2.waitKey(60)
    if k == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()