import numpy as np
import cv2
import json
from plugin import Plugin


class plugin(Plugin):
    angle = 0
    angle_min = 0
    angle_max = 0

    def __init__(self, param_string):
        params = json.loads(param_string)
        self.angle_min = params["angle"][0]
        self.angle_max = params["angle"][1]

    def prepare(self):
        self.angle = np.random.uniform(self.angle_min, self.angle_max)

    def augment_image(self, mat):
        if self.angle == 0:
            return mat

        cols = mat.shape[0]
        rows = mat.shape[1]
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2),
                                                  self.angle, 1.0)
        dst = cv2.warpAffine(mat, rotation_matrix, (cols, rows))

        return dst

    def augment_pixel_mask(self, mat):
        return self.augment_image(mat)

    def augment_depthmap(self, mat):
        return self.augment_image(mat)
