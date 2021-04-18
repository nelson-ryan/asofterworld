# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 12:24:39 2020

@author: xople

Testing the Google Cloud Vision API for OCR
"""

import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "nomadic-zoo-293819-8ccfdaa58681.json"

path = "comics/0702_graverobber.jpg"


#def detect_text(path):
"""Detects text in the file."""
from google.cloud import vision
# from google.cloud.vision import types # No longer necessary from version 2.0.0????
import io
client = vision.ImageAnnotatorClient()

with io.open(path, 'rb') as image_file:
    content = image_file.read()

image = vision.Image(content=content)

response = client.text_detection(image=image)
texts = response.text_annotations

#print(type(texts))

print('Texts:')

for text in texts:
    print('\n"{}"'.format(text.description))

#     vertices = (['({},{})'.format(vertex.x, vertex.y)
#                 for vertex in text.bounding_poly.vertices])

#     print('bounds: {}'.format(','.join(vertices)))

'''
Within 'texts' are objects for 'description', which is the read text,
and appears to be separated by word, and also includes new lines.
These are an EntityAnnotation type, but can be converted to str()

So I think I can just convert to string and return the result to the main function
'''

if response.error.message:
    raise Exception(
        '{}\nFor more info on error messages, check: '
        'https://cloud.google.com/apis/design/errors'.format(
            response.error.message))
        
#detect_text("comics/0001_myparents.jpg")