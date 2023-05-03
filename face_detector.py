import os
import abc
import cv2
import requests
import numpy as np
from json import JSONDecoder
from seetaface6.seetaface.api import *
from seetaface6.seetaface.face_struct import *


class Detector(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def load(self, model=None):
        pass

    @abc.abstractmethod
    def detect(self, image):
        pass


class CaffeModel(Detector):
    def __init__(self, model=None):
        if model is None:
            self.net = None
        else:
            self.load(model)

    def load(self, model=None):
        print("[INFO] loading caffe model...")
        protoPath = model[0]
        modelPath = model[1]
        self.net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
        print("[INFO] caffe model loaded")

    def detect(self, image):
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        try:
            assert detections[0, 0, 0, 2] > 0.5
            box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
        except (IndexError, AssertionError):
            print("[WARNING] no face detected")
            return -1, -1, -1, -1
        except Exception as e:
            print("[ERROR] {}".format(str(e)))
            return -1, -1, -1, -1
        (startX, startY, endX, endY) = box.astype("int")
        startX = max(0, startX)
        startY = max(0, startY)
        endX = min(w, endX)
        endY = min(h, endY)
        return startX, startY, endX, endY


class SeetaFace2(Detector):
    def __init__(self, model=None, landmark=False):
        if model is None:
            self.fd = None
            self.fl = None
        elif not landmark:
            self.load(model)
            self.fl = None
        else:
            self.landmark_init(model)

    def load(self, model=None):
        print("[INFO] loading seetaface detector...")
        self.fd = seetaface.FaceDetector(model)
        print("[INFO] seetaface detector loaded")

    def detect(self, image):
        image = seetaface.SeetaImage(image)
        faces = self.fd.detect(image)
        try:
            rect = faces[0].pos
        except IndexError:
            print("[WARNING] no face detected")
            return -1, -1, -1, -1
        except Exception as e:
            print("[Error] {}".format(str(e)))
            return -1, -1, -1, -1
        return rect.x, rect.y, rect.x + rect.width, rect.y + rect.height

    def landmark_init(self, model=None):
        print("[INFO] loading seetaface landmark detector...")
        self.load(model[0])
        self.fl = seetaface.FaceLandmarker(model[1])
        print("[INFO] seetaface landmark detector loaded")

    def landmark(self, image, boost=False):
        image_height = image.shape[0]
        image_width = image.shape[1]
        rect = seetaface.SeetaRect()
        if boost:
            rect.x, rect.y, rect.width, rect.height = 0, 0, image_width, image_height
        else:
            startX, startY, endX, endY = self.detect(image)
            rect.x, rect.y, rect.width, rect.height = startX, startY, endX - startX, endY - startY
        image = seetaface.SeetaImage(image)
        points = self.fl.detect(image, rect)
        kp = np.zeros([81, 2]).astype('int')
        for i in range(len(points)):
            kp[i, :] = np.array([points[i].y, points[i].x]).astype('int')
        return kp


class SeetaFace6(Detector):
    def __init__(self, landmark=False):
        if landmark:
            self.init_mask = FACE_DETECT | LANDMARKER68
        else:
            self.init_mask = FACE_DETECT
        self.seetaFace = None

    def load(self, model=None):
        print("[INFO] loading seetaface detector...")
        self.seetaFace = SeetaFace(self.init_mask)
        print("[INFO] seetaface detector loaded")

    def detect(self, image):
        faces = self.seetaFace.Detect(image)
        try:
            rect = faces.data[0].pos
        except IndexError:
            print("[WARNING] no face detected")
            return -1, -1, -1, -1
        except Exception as e:
            print("[Error] {}".format(str(e)))
            return -1, -1, -1, -1
        return rect.x, rect.y, rect.x + rect.width, rect.y + rect.height

    def landmark_init(self):
        print("[INFO] loading seetaface landmark detector...")
        self.init_mask = FACE_DETECT | LANDMARKER68
        self.seetaFace = SeetaFace(self.init_mask)
        print("[INFO] seetaface landmark detector loaded")

    def landmark(self, image, boost=False):
        image_height = image.shape[0]
        image_width = image.shape[1]
        rect = SeetaRect()
        if boost:
            rect.x, rect.y, rect.width, rect.height = 0, 0, image_width, image_height
        else:
            startX, startY, endX, endY = self.detect(image)
            rect.x, rect.y, rect.width, rect.height = startX, startY, endX - startX, endY - startY
        points = self.seetaFace.mark68(image, rect)
        kp = np.zeros([68, 2]).astype('int')
        for i in range(len(points)):
            kp[i, :] = np.array([points[i].y, points[i].x]).astype('int')
        return kp


class FacePP(Detector):
    def __init__(self, model=None):
        self.key = "xxx"
        self.secret = "xxx"
        if model is None:
            self.http_url = "https://api-cn.faceplusplus.com/facepp/v3/detect"
        else:
            self.load(model)

    def load(self, model=None):
        print("[INFO] loading face++ detector...")
        if model is not None:
            self.http_url = model
        print("[INFO] face++ detector loaded")

    def detect(self, image):
        if isinstance(image, np.ndarray):
            cv2.imwrite("temp_SeetaFace.png", image)
            image = "temp_SeetaFace.png"
        elif not isinstance(image, str):
            print("[ERROR] unsupported image format")
            return -1, -1, -1, -1
        data = {"api_key": self.key, "api_secret": self.secret}
        try:
            with open(image, "rb") as image_file:
                files = {"image_file": image_file}
                response = requests.post(self.http_url, data=data, files=files)
            req_con = response.content.decode('utf-8')
            req_dict = JSONDecoder().decode(req_con)
            face_dict = req_dict['faces'][0]
            startX = face_dict['face_rectangle']['left']
            startY = face_dict['face_rectangle']['top']
            width = face_dict['face_rectangle']['width']
            height = face_dict['face_rectangle']['height']
        except Exception as e:
            print("[ERROR] {}".format(str(e)))
            return -1, -1, -1, -1
        else:
            return startX, startY, startX + width, startY + height

def test_haar(image: np.ndarray):
    protoPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
    modelPath = os.path.sep.join(["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
    res10 = CaffeModel()
    res10.load([protoPath, modelPath])
    startX, startY, endX, endY = res10.detect(image)
    image_res10 = image.copy()
    cv2.rectangle(image_res10, (startX, startY), (endX, endY), (0, 0, 255), 2)
    cv2.imwrite('frame1_res10.png', image_res10)

def test_seetaface2(image: np.ndarray):
    model1Path = '/home/jason/databases/seetaface2_models/fd_2_00.dat'
    model2Path = '/home/jason/databases/seetaface2_models/pd_2_00_pts81.dat'
    modelPath = [model1Path, model2Path]
    seeta2 = SeetaFace2()
    seeta2.landmark_init(modelPath)
    startX, startY, endX, endY = seeta2.detect(image)
    key_points = seeta2.landmark(image)
    image_seeta2 = image.copy()
    cv2.rectangle(image_seeta2, (startX, startY), (endX, endY), (0, 0, 255), 2)
    cv2.imwrite('frame1_seeta2.png', image_seeta2)
    for key_point in key_points:
        cv2.circle(image_seeta2, (key_point[1], key_point[0]), 2, (0, 255, 0), 2)
    cv2.imwrite('frame1_landmark_seeta2.png', image_seeta2)

def test_seetaface6(image: np.ndarray):
    seeta6 = SeetaFace6()
    seeta6.landmark_init()
    startX, startY, endX, endY = seeta6.detect(image)
    key_points = seeta6.landmark(image)
    image_seeta6 = image.copy()
    cv2.rectangle(image_seeta6, (startX, startY), (endX, endY), (0, 0, 255), 2)
    cv2.imwrite('frame1_seeta6.png', image_seeta6)
    for ldmk_idx in range(key_points.shape[0]):
        key_point = key_points[ldmk_idx]
        cv2.putText(image_seeta6, str(ldmk_idx), (key_point[1], key_point[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.circle(image_seeta6, (key_point[1], key_point[0]), 2, (0, 255, 0), 2)
    cv2.imwrite('frame1_landmark_seeta6.png', image_seeta6)

def test_facepp(image: np.ndarray):
    facepp = FacePP()
    facepp.load()
    # startX, startY, endX, endY = facepp.detect(filepath)
    startX, startY, endX, endY = facepp.detect(image)
    image_facepp = image.copy()
    cv2.rectangle(image_facepp, (startX, startY), (endX, endY), (0, 0, 255), 2)
    cv2.imwrite('frame1_facepp.png', image_facepp)

