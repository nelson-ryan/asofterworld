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

    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations

    # Checking annotation output. This doesn't seem to include whitespace characters, but that shouldn't be an issue
    #print(type(texts))
    #print(texts[0])

    #print('Texts:')
    #print('\n"{}"'.format(texts[0].description))

    '''
    Within 'texts' are objects for 'description', which is the text that is read,
    and appears to be separated by word, and also includes new lines.
    These are an EntityAnnotation type, but can be converted to str()
    
    So I think I can just convert to string and return the result to the main function
    '''

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

    # or just return the annotation data like so, to compare bounding vertices outside the function:
    return texts


# Create cv2 contour list from texts coordinates, reference:
# https://stackoverflow.com/questions/14161331/creating-your-own-contour-in-opencv-using-python

def text2coords(ocr_output):
    word_contours = []  # for storing all words

    for text in ocr_output:
        word_vertices = []  # for storing all vertices for a single word
        # print(f"\n{text.description}")
        # print(format(text.bounding_poly.vertices))

        # TODO Change this to producing just a single vertex of the center of the
        #  box by using the average of the Xs and average of the Ys
        # Put each pair of vertices into a list pair and add to a list of vertices
        for vertex in text.bounding_poly.vertices:
            # Storing individual vertex coordinates for a word
            word_vertex = [vertex.x, vertex.y]
            word_vertices.append(word_vertex)
        # print(f"{text.bounding_poly.vertices[0].x}\t{text.bounding_poly.vertices[0].y}")

        # Convert list to numpy ndarray
        word_vertices = numpy.array(word_vertices, dtype=numpy.int32)
        word_contours.append(word_vertices)

    # Also convert final to ndarray
    word_contours = numpy.array(word_contours, dtype=numpy.int32)

    return word_contours

# test_path = "comics/0753_purina.jpg"
# texts = detect_text(test_path)
# print(text2coords(texts))


