# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 09:53:37 2020

@author: nelsonr
"""

# bulk of code from https://stackoverflow.com/a/56473372

# solution to ordering modified from https://stackoverflow.com/a/39445901


import cv2 # OpenCV (opencv-python)

# Load image
im = cv2.imread('comics/0392_sit.jpg',cv2.IMREAD_UNCHANGED)

# Create greyscale version
gr = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# Threshold to get black and white
_,grthresh = cv2.threshold(gr,230,255,cv2.THRESH_BINARY)
cv2.imwrite('result-1.png',grthresh)

# Median filter to remove JPEG noise
grthresh = cv2.medianBlur(grthresh,11)
cv2.imwrite('result-2.png', grthresh)

# Find contours
contours, hierarchy = cv2.findContours(grthresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


def get_contour_precedence(contour, cols):
    tolerance_factor = 1
    origin = cv2.boundingRect(contour)
    return (origin[0] // tolerance_factor) * cols + origin[0]


contours.sort(key=lambda y: get_contour_precedence(y, im.shape[0]))

# Look through contours, checking what we found
frame = 0
for i in range(len(contours)):
    area = cv2.contourArea(contours[i])
    # Only consider ones taller than around 100 pixels and wider than about 300 pixels
    if area > 30000:
        # Get cropping box and crop
        rc = cv2.minAreaRect(contours[i])
        box = cv2.boxPoints(rc)
        Xs = [box[0, 0], box[1, 0], box[2, 0], box[3, 0]]
        Ys = [box[0, 1], box[1, 1], box[2, 1], box[3, 1]]
        x0 = int(round(min(Xs)))
        x1 = int(round(max(Xs)))
        y0 = int(round(min(Ys)))
        y1 = int(round(max(Ys)))
        cv2.imwrite(f'frame-{frame}.png', im[y0:y1, x0:x1])
        frame += 1
