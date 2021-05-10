# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 09:53:37 2020

@author: nelsonr
"""

# bulk of code from https://stackoverflow.com/a/56473372
# solution to ordering from https://stackoverflow.com/a/39445901

import cv2  # OpenCV (opencv-python)
import os


# def split_frames(filename, dest_folder="split-frames/"):
import cv2.cv2


def split_frames(filename):
    # if not os.path.exists(dest_folder):
    #     os.mkdir(dest_folder)
    # title = filename.split(sep="/")[-1].split(sep=".")[-2]

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

    # From https://stackoverflow.com/a/56473372 again
    # Look through contours, checking what we found
    frame = 0
    frame_bounds = {}
    contour_shortlist = []
    for i in range(1, len(contours)):
        area = cv2.contourArea(contours[i])
        # Only consider ones taller than around 100 pixels and wider than about 300 pixels
        if area > 30000:
            contour_shortlist.append(contours[i])
            # Get cropping box and crop
            rc = cv2.minAreaRect(contours[i])
            box = cv2.boxPoints(rc)
            Xs = [box[0, 0], box[1, 0], box[2, 0], box[3, 0]]
            Ys = [box[0, 1], box[1, 1], box[2, 1], box[3, 1]]
            x0 = int(round(min(Xs)))
            x1 = int(round(max(Xs)))
            y0 = int(round(min(Ys)))
            y1 = int(round(max(Ys)))
            # No longer save the files
            # cv2.imwrite(f'split-test/{title}-{frame:02d}.png', im[y0:y1, x0:x1])
            frame_id = 'frame_' + str(frame)
            # print(frame_id)
            # print(type(box))
            # frame_bounds[frame_id] = [x0, x1, y0, y1]
            frame_bounds[frame_id] = box
            frame += 1
    # Rather than save each individual frame and running those each through the Cloud Vision OCR,
    # I may be able to just compare these bounding boxes to the bounding polys from the OCR to determine
    # which frame they're in (cutting the number of API requests down by more than a factor of 3).

    print(type(contour_shortlist))
    print(contour_shortlist)

    # TODO reference for creating contour from coordinates:
    #  https://stackoverflow.com/questions/14161331/creating-your-own-contour-in-opencv-using-python

    # Visual testing of contour placement
    cv2.drawContours(im, contour_shortlist, -1, (0, 255, 0), 3)
    cv2.waitKey(0)
    cv2.imshow('Contours', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # TODO Return new contour list instead of frame bounds
    # return frame_bounds
    return contour_shortlist
    # TODO Add function to create contour list from OCR output

#split_frames('comics/0753_purina.jpg')
