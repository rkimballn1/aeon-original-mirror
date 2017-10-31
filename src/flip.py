import numpy as np
import cv2

def execute(mat):
    dst = cv2.flip(mat, 1)
    return dst
