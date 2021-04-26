# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 09:53:37 2020

@author: nelsonr
"""

# bulk of code from https://stackoverflow.com/a/56473372
# solution to ordering from https://stackoverflow.com/a/39445901

import cv2  # OpenCV (opencv-python)
import os


def split_frames(filename, dest_folder="split-frames/"):
    if not os.path.exists(dest_folder):
        os.mkdir(dest_folder)

    title = filename.split(sep="/")[-1].split(sep=".")[-2]

    # Load image
    im = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

    # Create greyscale version
    gr = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # Threshold to get black and white
    _, grthresh = cv2.threshold(gr, 230, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, hierarchy = cv2.findContours(grthresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # This function comes from https://stackoverflow.com/a/39445901
    def get_contour_precedence(contour, cols):
        tolerance_factor = 50
        origin = cv2.boundingRect(contour)
        return (origin[1] // tolerance_factor) * cols + origin[0]

    contours.sort(key=lambda y: get_contour_precedence(y, im.shape[0]))

    # From https://stackoverflow.com/a/56473372 again
    # Look through contours, checking what we found
    frame = 0
    for i in range(1, len(contours)):
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
            cv2.imwrite(f'split-test/{title}-{frame:02d}.png', im[y0:y1, x0:x1])
            frame += 1


split_frames('comics/0753_purina.jpg')