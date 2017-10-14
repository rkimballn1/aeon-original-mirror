import numpy as np
import cv2

def execute(mat):
    print "Flip"
    
    dst = cv2.flip(mat, 1)

    print "Flipped"
    return dst

