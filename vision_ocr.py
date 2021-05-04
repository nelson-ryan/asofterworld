# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 12:24:39 2020

@author: nelsonr

Testing the Google Cloud Vision API for OCR
"""

import os
from google.cloud import vision
import io

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "nomadic-zoo-293819-8ccfdaa58681.json"

testpath = "comics/0753_purina.jpg"


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

print(detect_text(testpath))