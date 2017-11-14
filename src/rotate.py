import numpy as np
import cv2
from plugin import Plugin
import json
import random


class plugin(Plugin):
    angle = 0
    angle_min = 0
    angle_max = 0

    def __init__(self, param_string):
        params = json.loads(param_string)
        self.angle_min = params["angle"][0]
        self.angle_max = params["angle"][1]

    def prepare(self):
        angle = random.randint(self.angle_min, self.angle_max)

    def augment_image(self, mat):
        if angle == 0:
            return mat

        cols, rows = mat.shape()
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle,
                                                  1.0)
        dst = cv2.warpAffine(mat, rotation_matrix, (cols, rows))

        #print dst
        return dst


#img = cv2.imread("dickbutt.jpg", 0)
#new = execute(img, 45)
#cv2.imwrite("dickbutt2.jpg", new)
