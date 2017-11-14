import numpy as np
import cv2
import json
from plugin import Plugin


class plugin(Plugin):
    probability = 0.5
    do_flip = False

    def __init__(self, param_string):
        if len(param_string)>0:
            params = json.loads(param_string)
            self.probability = params["probability"]

    def prepare(self):
        self.do_flip = np.random.uniform() < self.probability

    def augment_image(self, mat):
        if self.do_flip:
            dst = cv2.flip(mat, 1)
        else:
            dst = mat
        return dst
