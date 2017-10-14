import numpy as np
import cv2

def execute(mat, angle):
    if angle == 0:
        return mat
    
    print "Rotate"
    rows, cols = mat.shape

    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1.0)
    dst = cv2.warpAffine(mat, rotation_matrix, (cols, rows))

    print "Rotated"
    return dst

#img = cv2.imread("dickbutt.jpg", 0)
#new = execute(img, 45)
#cv2.imwrite("dickbutt2.jpg", new)
