# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 09:53:37 2020

@author: nelsonr
"""

# bulk of code from https://stackoverflow.com/a/56473372
# solution to ordering from https://stackoverflow.com/a/39445901

import cv2  # OpenCV (opencv-python)
import numpy


def find_frames(filename):
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

    contours.sort(key=lambda x: get_contour_precedence(x, im.shape[0]))

    # From https://stackoverflow.com/a/56473372 again (not really applicable anymore)
    # Look through contours, checking what we found
    contour_shortlist = []
    # Include only contours with sufficient area
    for i in range(1, len(contours)):
        area = cv2.contourArea(contours[i])
        # Only consider ones taller than around 100 pixels and wider than about 300 pixels
        if area > 30000:
            box = cv2.minAreaRect(contours[i])  # Get minimal points instead of all
            box = cv2.boxPoints(box)  # Converts tuple to contour ndarray
            contour_shortlist.append(box)
    # Convert entire list to contour array
    contour_shortlist = numpy.array(contour_shortlist, dtype=numpy.int32)

    return contour_shortlist
