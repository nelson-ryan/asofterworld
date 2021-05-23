# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 11:21:20 2020
@author: nelson-ryan

"""

import io
import os  # for directory checking and creation
import requests
import bs4  # beautifulsoup4
import cv2
import numpy as np
from google.cloud import vision
import json

# Google Vision credential
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "nomadic-zoo-293819-8ccfdaa58681.json"

# UPDATE FIRST COMIC NUMBER IN VARIABLE DECLARATION
def main():
    first_comic = 1246
    # TODO Check for both comic data and image file to determine whether to pull either
    #  (presently it does not download image if data is present) 

    # If json data already exists, load it; else, initialize for pull from web
    if os.path.exists('comics.json'):
        with open('comics.json', 'r') as read_file:
            comics = json.load(read_file)
    else:
        comics = {}
    
    # For each comic in range, first check if data already present before pull
    pulling_comic = first_comic
    while True:
        comicdictkey = f'comic_{pulling_comic}'
        if comicdictkey not in comics:
            retrieved_comic = save_comic(pulling_comic)
            if retrieved_comic:
                comics[comicdictkey] = retrieved_comic
                print(f'Grabbed #{pulling_comic}')
            else:
                break
        else:
            print(f'Skipped #{pulling_comic}')
        pulling_comic += 1

    # Iterate through each saved comic to read in OCR
    for comic in comics:
        if "frame_text" not in comics[comic]:
            comic_path = comics[comic].get("save_loc")
            
            frame_contours = find_frames(comic_path)
            ocr_text = detect_text(comic_path)
            ocr_contours, ocr_points = text2coords(ocr_text)
            '''All of these ocr_* things are convoluted; this might be an ideal place to use a class'''

            # drawTest(comic_path, frame_contours, ocr_contours, ocr_points)

            text_by_frame = group_frame_text(frames=frame_contours,
                                            text_points=ocr_points,
                                            text=ocr_text)
            comics[comic]["frame_text"] = text_by_frame
            print(f'Ran OCR for {comic}')
        else:
            print(f'Skipped OCR for {comic}')

    with open('comics.json', 'w') as write_file:
        json.dump(comics, write_file)

# Get individual comic info and save it to a dictionary
def save_comic(n, save_dest_folder='comics'):
    comic_dict = {}
    comic_number = n

    res = requests.get(f'https://www.asofterworld.com/index.php?id={comic_number}')
    res.raise_for_status()
    soup = bs4.BeautifulSoup(res.text, "html.parser")
    comic = soup.select("#comicimg > img")
    url = comic[0].get('src')
    filename = url.split('/')[-1]  # filename is last part of URL, following the last slash
    
    if not filename:
        return # If there's no filename, there's no comic, so stop
    alt_text = comic[0].get('title')

    img_url = requests.get(url)

    if not os.path.exists(f'{save_dest_folder}/'):
        os.mkdir(f'{save_dest_folder}/')
    save_loc = f'{save_dest_folder}/{comic_number:04d}_{filename}'
    # no need to save it if it's already there
    if not os.path.exists(save_loc):
        with open(save_loc, 'wb') as img:
            img.write(img_url.content)

    # These could well also be class attributes
    comic_dict["comic_number"] = comic_number
    comic_dict["filename"] = filename
    comic_dict["alt_text"] = alt_text
    comic_dict["save_loc"] = save_loc

    return comic_dict

def find_frames(filename):
    # Load image
    im = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    
    # Create greyscale version--only if it has a color layer (not already only grey)
    if len(im.shape) == 3:  # shape is rows, columns, and channels (if color)
        gr = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    elif len(im.shape) == 2:  # shape as no channels count if no color
        gr = im
    # Threshold to get black and white
    _, grthresh = cv2.threshold(gr, 230, 255, cv2.THRESH_BINARY)
    # Find contours
    contours, _ = cv2.findContours(grthresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # This function comes from https://stackoverflow.com/a/39445901
    def get_contour_precedence(contour, cols):
        tolerance_factor = 50
        origin = cv2.boundingRect(contour)
        return (origin[1] // tolerance_factor) * cols + origin[0]

    contours.sort(key=lambda x: get_contour_precedence(x, im.shape[0]))

    # Basis from https://stackoverflow.com/a/56473372 (now heavily gutted, though):
    # Create empty list for contours of frames
    contour_shortlist = []
    # Identify frames by including only contours with sufficient area
    for i in range(1, len(contours)):
        area = cv2.contourArea(contours[i])
        # TODO Test if this threshold leaves any false negatives/positives by checking
        #  1) frames are multiples of three and 2) all OCR words are in exactly one frame
        if area > 50000:
            box = cv2.minAreaRect(contours[i])  # Get minimal points instead of all
            box = cv2.boxPoints(box)  # Converts tuple to contour ndarray
            contour_shortlist.append(box)
    # Convert entire list to contour ndarray
    contour_shortlist = np.array(contour_shortlist, dtype=np.int32)

    return contour_shortlist


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


def text2coords(ocr_output):
    # Create cv2 contour list from texts coordinates, reference:
    # https://stackoverflow.com/questions/14161331/creating-your-own-contour-in-opencv-using-python

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
        # Convert list to contour ndarray
        word_vertices = np.array(word_vertices, dtype=np.int32)
        word_contours.append(word_vertices)
        word_points.append(point)

    # Also convert final to contour ndarray
    word_contours = np.array(word_contours, dtype=np.int32)

    return word_contours, word_points


def drawTest(comic_path, frame_contours, ocr_contours, ocr_points):
    # Testing that drawContour successfully places both contour groups
    img = cv2.imread(comic_path, cv2.IMREAD_UNCHANGED)
    cv2.drawContours(img, frame_contours, -1, (255, 255, 0), 2)
    cv2.drawContours(img, ocr_contours, -1, (255, 0, 255), 2)
    for point in ocr_points:
        cv2.circle(img, tuple(point), radius=3, color=(0, 255, 0), thickness=3)
    cv2.imshow('circle', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Check for word location within frame contours, add corresponding text accordingly
def group_frame_text(frames, text_points, text):
    text_by_frame = []
    for j in range(len(frames)):
        text_by_frame.append([])
        for k in range(1, len(text_points)):
            # Check if text is inside frame; pointPolygonTest returns 1 if yes
            if cv2.pointPolygonTest(contour=frames[j],
                                    pt=tuple(text_points[k]),
                                    measureDist=False) > 0:
                text_by_frame[j].append(text[k].description)
        # Join separate list items into a single string
        text_by_frame[j] = ' '.join(text_by_frame[j])
    # get rid of empty strings for frames without text
    text_by_frame[:] = [x for x in text_by_frame if x != '']
    return text_by_frame


main()
