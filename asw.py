"""
Created on Tue Oct 20 11:21:20 2020
@author: nelson-ryan
"""
# spaCy
# https://stackoverflow.com/questions/56896753/is-there-a-way-to-get-entire-constituents-using-spacy
# Stanza
# https://stanfordnlp.github.io/stanza/constituency.html

# TODO: Reorganize/move each step outside of constructor, allowing individual
#       elements to be re-loaded as needed
# TODO: Fix contours so that 'asofterworld.com' isn't included in final panel
# TODO: Use Stanza (https://stanfordnlp.github.io/stanza/constituency.html) to
#       parse into syntactic consituents
# TODO: Compare Stanza constituents to comic frame boundaries.
# TODO: Quantify
# TODO: Visualize

import requests
import io
import os
import bs4
import cv2
import numpy as np
import json
from pathlib import Path
from google.cloud import vision

FIRST = 1
LAST = 1249 # (1248 comics, non-inclusive range)
BASE_URL = 'https://www.asofterworld.com/index.php?id='

# Google Vision credential
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
    "nomadic-zoo-293819-43dc8cc8b69f.json")
save_dest_folder = Path('comics')

class Comic:
    """ASofterWorld individual comic object."""
    def __init__(self, number):

        self.number = number
        res = requests.get(BASE_URL + str(self.number))
        # raise_for_status raises Exception if site can't be reached, but this
        # is not an indicator of the presence of a comic, as the site redirects
        res.raise_for_status()
        soup = bs4.BeautifulSoup(res.text, "html.parser")
        comic = soup.select("#comicimg > img")
        self.url = comic[0].get('src')
        self.alt_text = comic[0].get('title')
        self.filename = self.url.split('/')[-1]
        self.save_loc = self.save_comic()
        self.frame_contours = self.find_frames()
        self.ocr_text = self.detect_text()
        self.ocr_contours, self.ocr_points = self.text2coords()
        self.frame_text = self.group_frame_text()

    def save_comic(self):

        # no need to save it if it's already there
        save_loc = (Path(save_dest_folder) /
                    f'{self.number:04d}_{self.filename}')
        if save_loc.exists():
            print(f'{save_loc} already exists')
        else:
            if not save_dest_folder.exists():
                save_dest_folder.mkdir()
            with open(save_loc, 'wb') as img:
                img_url = requests.get(self.url)
                img.write(img_url.content)
        return save_loc

    ### OCR ###

    def find_frames(self):

        # Load locally-saved image
        im = cv2.imread(str(self.save_loc), cv2.IMREAD_UNCHANGED)

        # Check if image is missing final image data (as is the case with #363,
        # which causes errors with OCR, so this may be better to do there...)
        # Solution adapted from https://stackoverflow.com/a/68918602/12662447
        with open(self.save_loc, 'rb') as imgopen:
            imgopen.seek(-2,2)
            # If so, overwrite and re-read image file
            if imgopen.read() != b'\xff\xd9':
                cv2.imwrite(str(self.save_loc), im)
                im = cv2.imread(str(self.save_loc), cv2.IMREAD_UNCHANGED)


        # Create greyscale version
        # IF it has a color layer (i.e. not already only grey)
        if len(im.shape) == 3: # shape is rows, columns, channels (if color)
            gr = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        elif len(im.shape) == 2:  # shape has no 'channels' value if not color
            gr = im
        # Threshold to get black and white
        _, grthresh = cv2.threshold(gr, 230, 255, cv2.THRESH_BINARY)
        # Find contours
        contours, _ = cv2.findContours(grthresh,
                                       cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # This function comes from https://stackoverflow.com/a/39445901
        def get_contour_precedence(contour, cols):
            tolerance_factor = 50
            origin = cv2.boundingRect(contour)
            return (origin[1] // tolerance_factor) * cols + origin[0]

        contours = list(contours)
        contours.sort(key=lambda x: get_contour_precedence(x, im.shape[0]))

        # Heavily gutted from https://stackoverflow.com/a/56473372
        # Create empty list for contours of frames
        contour_shortlist = []
        # Identify frames by including only contours with sufficient area
        for i in range(1, len(contours)):
            area = cv2.contourArea(contours[i])
            # TODO Test if this threshold leaves any false negatives/positives:
            #  1) check whether frames are multiples of three and
            #  2) check that all OCR words are in exactly one frame
            if area > 50000:
                # Get minimal points instead of all
                box = cv2.minAreaRect(contours[i])
                # Convert tuple to contour ndarray
                box = cv2.boxPoints(box)
                # Add to output
                contour_shortlist.append(box)
        # Convert entire list to contour ndarray
        contour_shortlist = np.array(contour_shortlist, dtype=np.int32)

        return contour_shortlist

    def detect_text(self):
        """Detects text in the file."""

        with io.open(self.save_loc, 'rb') as image_file:
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

    def text2coords(self):
        # Create cv2 contour list from texts coordinates, reference:
        # https://stackoverflow.com/questions/14161331/
        word_contours = []  # for storing all words
        word_points = []

        for text in self.ocr_text:
            word_vertices = []  # for storing all vertices for a single word
            word_Xs = []
            word_Ys = []

            # Put each vertex pair into a list pair & add to a list of vertices
            for vertex in text.bounding_poly.vertices:
                # Storing individual vertex coordinates for a word
                word_vertex = [vertex.x, vertex.y]
                word_vertices.append(word_vertex)
                word_Xs.append(vertex.x)
                word_Ys.append(vertex.y)
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

    # Check for word location within frame contours, add corresponding text
    def group_frame_text(self):
        text_by_frame = []
        for j in range(len(self.frame_contours)):
            text_by_frame.append([])
            for k in range(1, len(self.ocr_points)):
                # Check if text is inside frame; pointPolygonTest 1 if yes
                if cv2.pointPolygonTest(contour=self.frame_contours[j],
                                        pt=tuple(self.ocr_points[k]),
                                        measureDist=False) > 0:
                    text_by_frame[j].append(self.ocr_text[k].description)
            # Join separate list items into a single string
            text_by_frame[j] = ' '.join(text_by_frame[j])
        # get rid of empty strings for frames without text
        text_by_frame[:] = [x for x in text_by_frame if x != '']
        return text_by_frame

    def saveContourImage(self):
        # Testing that drawContour successfully places both contour groups
        img = cv2.imread(str(self.save_loc), cv2.IMREAD_UNCHANGED)
        cv2.drawContours(img, self.frame_contours, -1, (255, 255, 0), 2)
        cv2.drawContours(img, self.ocr_contours, -1, (255, 0, 255), 2)
        for point in self.ocr_points:
            cv2.circle(img,
                       tuple(point),
                       radius=2,
                       color=(0, 255, 0),
                       thickness=3
            )
        # cv2.imshow('circle', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        save_loc = (Path(save_dest_folder) /
                    f'{self.number:04d}_{self.filename.split(".")[0]}_contours'
                    '.jpg'
                    )
        cv2.imwrite(str(save_loc), img)


if __name__ == '__main__':


    # If json data already exists, load it
    jsonfile = 'comics.json'
    if Path(jsonfile).is_file():
        with open(jsonfile, 'r') as read_file:
            comicsjson = json.load(read_file)
    else:
        comicsjson = {}

    comics = []

    # Iterate through range of comic numbers
    for i in range(FIRST, LAST):
        comicdictkey = f'comic_{i}'

        if comicdictkey not in comicsjson:
            comics.append(Comic(i))
        # if basic info not in json entry:
            # add basic info
            # comicsjson[comicdictkey][comic_number] = comics[-1].number
            # comicsjson[comicdictkey][alt_text] = comics[-1].alt_text
        # if original image not saved:
            # download and save image
        # if contours not recorded:
            # find and record contours
        # if OCR text not recorded:
            # read OCR
        # if panel-specific text not identified:
            # identify and save panel-specific text
            comicsjson[comicdictkey] = {'alt_text' : comics[-1].alt_text,
                                        'comic_number' : comics[-1].number,
                                        'frame_text' : comics[-1].frame_text,
                                        'save_loc' : str(comics[-1].save_loc)}
            comics[-1].saveContourImage()
        else:
            print(f'{comicdictkey} already recorded')


    with open('comics.json', 'w') as write_file:
        json.dump(comicsjson, write_file, sort_keys=True)
