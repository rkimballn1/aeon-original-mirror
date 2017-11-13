from plugin import Plugin
import numpy as np
import cv2


class plugin(Plugin):
    def __init__(self, param_string):
        print("# flip constructor called with "+param_string)
        pass

    def prepare(self):
        print("# flip prepare called")
        pass

    def augment_image(self, mat):
        print("# flip augment_image called with mat of size "+str(mat.size))
        dst = cv2.flip(mat, 1)
        return dst
