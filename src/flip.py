import numpy as np
import cv2
import json
from plugin import Plugin


class plugin(Plugin):
    probability = 0.5
    do_flip = False
    width = 0

    def __init__(self, param_string):
        if len(param_string) > 0:
            params = json.loads(param_string)
            if params.has_key("probability"):
                self.probability = params["probability"]
            if params.has_key("width"):
                self.width = params["width"]
            else:
                raise KeyError('width required for flip.py')

    def prepare(self):
        print "prepare"
        self.do_flip = np.random.uniform() < self.probability

    def augment_image(self, mat):
        print "augment_image"
        if self.do_flip:
            dst = cv2.flip(mat, 1)
        else:
            dst = mat
        return dst

    def augment_boundingbox(self, boxes):
        print "augment_boundingbox"
        if self.do_flip:
            for i in xrange(len(boxes)):
                xmax = boxes[i]["xmax"]
                boxes[i]["xmax"] = self.width - boxes[i]["xmin"] - 1
                boxes[i]["xmin"] = self.width - xmax - 1
        return boxes

    def augment_pixel_mask(self, mat):
        return self.augment_image(mat)

    def augment_depthmap(self, mat):
        return self.augment_image(mat)
