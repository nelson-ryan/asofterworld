# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 12:24:39 2020

@author: nelson-ryan
"""

import os
from google.cloud import vision
import io
import numpy

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "nomadic-zoo-293819-8ccfdaa58681.json"


def detect_text(path):
    """Detects text in the file."""

    with io.open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)

    client = vision.ImageAnnotatorClient()
    response = client.text_detection(image=image)
    texts = response.text_annotations
    #print('\n"{}"'.format(texts[0].description))
    # Keeping this print line here to reference the description object

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

    return texts


# Create cv2 contour list from texts coordinates, reference:
# https://stackoverflow.com/questions/14161331/creating-your-own-contour-in-opencv-using-python

def text2coords(ocr_output):
    word_contours = []  # for storing all words
    word_points = []

    for text in ocr_output:
        word_vertices = []  # for storing all vertices for a single word
        word_Xs = []
        word_Ys = []
        # print(f"\n{text.description}")
        # print(format(text.bounding_poly.vertices))

        # Put each pair of vertices into a list pair and add to a list of vertices
        for vertex in text.bounding_poly.vertices:
            # Storing individual vertex coordinates for a word
            word_vertex = [vertex.x, vertex.y]
            word_vertices.append(word_vertex)
            word_Xs.append(vertex.x)
            word_Ys.append(vertex.y)
        # print(f"{text.bounding_poly.vertices[0].x}\t{text.bounding_poly.vertices[0].y}")
        point_X = int(sum(word_Xs)/len(word_Xs))
        point_Y = int(sum(word_Ys)/len(word_Ys))
        point = [point_X, point_Y]
        # Convert list to numpy ndarray
        word_vertices = numpy.array(word_vertices, dtype=numpy.int32)
        word_contours.append(word_vertices)
        word_points.append(point)

    # Also convert final to ndarray
    word_contours = numpy.array(word_contours, dtype=numpy.int32)

    return word_contours, word_points



# t = detect_text("comics/1248_ruby.jpg")
# print(type(text2coords(t)))
# for i in text2coords(t):
#     print(i)