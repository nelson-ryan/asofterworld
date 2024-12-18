"""
Created on Tue Oct 20 11:21:20 2020
@author: nelson-ryan
"""
# spaCy
# https://stackoverflow.com/questions/56896753/
# Stanza
# https://stanfordnlp.github.io/stanza/constituency.html

# TODO: Use Stanza (https://stanfordnlp.github.io/stanza/constituency.html) to
#       parse into syntactic consituents
#   1 get flat string from comic
#   2 feed flat string into Stanza --> should provide boundaries
"""
nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
doc = nlp('This is a test')
for sentence in doc.sentences:
    print(sentence.constituency)
"""
# TODO: Compare Stanza constituents to comic frame boundaries.
# TODO: Quantify
# TODO: Visualize
import timeit
start = timeit.default_timer()

import requests
import io
import os
import bs4
import cv2
import numpy as np
import json
from pathlib import Path
from google.cloud import vision
import stanza
from typing import Sequence

FIRST = 1
LAST = 1249 # (1248 comics, non-inclusive range)
JSONFILE = 'comics.json'

# Google Vision credential
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
    "nomadic-zoo-293819-eecd00e68b8e.json"
)

class Comic:
    """
    ASofterWorld individual comic object.
    """
    NLP = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
    BASE_URL = 'https://www.asofterworld.com/index.php?id='

    IMG_FOLDER = 'comics'
    SAVE_DEST_FOLDER = Path(IMG_FOLDER)
    CLIENT = vision.ImageAnnotatorClient()

    def __init__(self, number: int):
        """
        """
        self.id: int = number
        self.url: str = self.BASE_URL + str(self.id)
        self.soup: bs4.element.Tag = self._get_soup()
        try:
            self.img_url: str = str(self.soup.get('src'))
        except IndexError:
            raise Exception(f'Comic {self.id} does not exist')
        self.alt_text: str = str(self.soup.get('title'))
        self.filename: str = self.img_url.split('/')[-1]
        self.local_img_path: Path = (
            Path(self.SAVE_DEST_FOLDER) / f'{self.id:04d}_{self.filename}'
        )

        # 
        self.img: np.ndarray = np.array(None)

        self.panel_contours = []
        self.textboxes = []
        self.ocr_text: Sequence = []
        self.ocr_contours: Sequence = []
        self.ocr_points: Sequence = []
        self.panel_text: list = []
        self.sentences = None # added by self.parse(), this shows a parsed
                              # constituency, though in a form that likely
                              # isn't immediately useful, but promising


    def process_comic(self):
        """
        Full sequence for each comic.
        """
        self._download_jpg()
        self.img = cv2.imread(str(self.local_img_path))
        self.img_grey = (
            self.img.copy() if len(self.img.shape) == 2
            else cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        )
        self.img_contoured = self.img.copy()
        self._fix_broken_jpg()
        self.ocr_text = self._read_text()
        self.panel_contours = self._find_panel_contours()
        # self.show_img("frames")
        self.textboxes = self._find_textboxes()
        # self.show_img("text boxes")
        self.panel_text = self._assign_panel_text()
        # self.show_img("text points")
        return


    def show_img(self, label = "img") -> None:
        """
        Display whichever image in a temporary window.
        """
        cv2.imshow(label, self.img_contoured)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return None


    def _get_soup(self) -> bs4.element.Tag:
        """
        Retrieve remote html from which to pull info.
        """
        res = requests.get(self.url)
        res.raise_for_status()
        soup_element_tag = (
            bs4.BeautifulSoup(res.text, "html.parser")
            .select("#comicimg > img")
        )[0]
        return soup_element_tag


    def _download_jpg(self):
        """
        Exactly what it says on the tin,
        if the image isn't already saved locally
        """
        if self.local_img_path.exists():
            print(f'{self.local_img_path} already exists. Skipping download.')
        else:
            if not self.SAVE_DEST_FOLDER.exists():
                self.SAVE_DEST_FOLDER.mkdir()
            with open(self.local_img_path, 'wb') as img:
                img_url = requests.get(self.img_url)
                img.write(img_url.content)
                print("Saved " + str(self.local_img_path))


    def _fix_broken_jpg(self):
        """
        Check if image is missing file-final image data
        (as is the case with #363)
        Solution adapted from https://stackoverflow.com/a/68918602/12662447
        """
        with open(self.local_img_path, 'rb') as imgfile:
            imgfile.seek(-2,2)
            # If end of file is different than expected, overwrite
            if imgfile.read() != b'\xff\xd9':
                cv2.imwrite(str(self.local_img_path), self.img)


    @staticmethod
    def _get_contour_precedence(contour, cols):
        """
        Used for sorting panel frames wequentially, especially those that
        don't fall neatly in the Cartesian plane when there are more than three
        panels.
        This function comes from https://stackoverflow.com/a/39445901
        """
        tolerance_factor = 50
        origin = cv2.boundingRect(contour)
        return (origin[1] // tolerance_factor) * cols + origin[0]


    def _find_panel_contours(
        self,
        save_thresh: bool = False,
        save_blur = False
    ) -> np.ndarray:
        """
        Identify the boundaries of individual panels
        """

        # Threshold to get binary black-and-white
        _, framethresh = cv2.threshold(self.img_grey, 40, 255, cv2.THRESH_BINARY)

        # Check threshold result by saving image
        if save_thresh:
            cv2.imwrite(
                filename = (
                    f'comics/{self.id:04d}_{self.filename.split(".")[0]}_textthresh.jpg'
                ),
                img = framethresh
            )
        # Median filter to remove jpg noise
        framethresh = cv2.medianBlur(framethresh, 3)
        if save_blur:
            cv2.imwrite(
                filename = (
                    f'{self.id:04d}_{self.filename.split(".")[0]}_blur.jpg'
                ),
                img = framethresh
            )

        # Find contours
        contours, _ = cv2.findContours(
            image = framethresh,
            mode = cv2.RETR_LIST,
            method = cv2.CHAIN_APPROX_SIMPLE
        )

        contours = list(contours)
        contours.sort(
            key = lambda x: self._get_contour_precedence(x, self.img.shape[0])
        )

        # Heavily gutted from https://stackoverflow.com/a/56473372
        # Create empty list for contours of panels
        panel_contours = []
        # Identify panels by including only contours with sufficient area

        for i in range(1, len(contours)):
            area = cv2.contourArea(contours[i])

            # TODO Test if this threshold leaves any false negatives/positives:
            #  1) check whether panels are multiples of three and
            #  2) check that all OCR words are in exactly one panel

            if area > 50000:
                # Get minimal points instead of all
                box = cv2.minAreaRect(contours[i])
                # Convert tuple to contour ndarray
                box = cv2.boxPoints(box)
                # Add to output
                panel_contours.append(box)
        # Convert entire list to contour ndarray
        panel_contours = np.array(panel_contours, dtype=np.int32)

        return panel_contours


    def _find_textboxes(self):
        """I think the goal here is to better identify actual comic text
        versus anything in the photo
        This copies much from _find_panel_contours
        Seems to work for comic #8
        """

        _, textboxthresh = cv2.threshold(
            src = self.img_grey,
            thresh = 240,
            maxval = 255,
            type = cv2.THRESH_BINARY
        )
        contours, _ = cv2.findContours(
            image = textboxthresh,
            mode = cv2.RETR_LIST,
            method = cv2.CHAIN_APPROX_SIMPLE
        )
        self.img_tbt = textboxthresh

        # Identify panels by including only contours with sufficient area
        textbox_contours = []
        textbox_contours = [
            cv2.boxPoints(cv2.minAreaRect(c)) for c in contours
            if 200 < cv2.contourArea(c) < 50000
        ]
        # Convert entire list to contour ndarray
        textbox_contours = np.array(textbox_contours, dtype=np.int32)

        cv2.imwrite(
            f'comics/{self.id:04d}_{self.filename.split(".")[0]}_textthresh.jpg',
            textboxthresh
        )

        self.draw_textbox_outlines()

        return textbox_contours


    def _read_text(self) -> Sequence:
        """
        Use Google Vision OCR to read text in the image.
        """

        with io.open(self.local_img_path, 'rb') as image_file:
            content = image_file.read()
        image = vision.Image(content = content)

        response = Comic.CLIENT.text_detection(image = image)
        texts = response.text_annotations
        #print('\n"{}"'.format(texts[0].description))
        # Keeping this print line here to reference the description object

        if response.error.message:
            raise Exception(
                '{}\nFor more info on error messages, check: '
                'https://cloud.google.com/apis/design/errors'.format(
                    response.error.message))

        return texts


    def _text2coords(self):
        """
        Create cv2 contour list from texts coordinates,
        Called by _assign_panel_text()
        Defines values for self.ocr_contours and self.ocr_points

        Reference: https://stackoverflow.com/questions/14161331/
        """
        word_contours = []  # for storing all words
        word_points = []

        for text in self.ocr_text[1:]:
            word_vertices = []  # for storing all vertices for a single word
            word_Xs = []
            word_Ys = []

            # convert to center point?
            # Put each vertex pair into a list pair & add to a list of vertices
            for vertex in text.bounding_poly.vertices:

                # Storing individual vertex coordinates for a word
                word_vertices.append([vertex.x, vertex.y])

                word_Xs.append(vertex.x)
                word_Ys.append(vertex.y)
            # Convert list to contour ndarray
            word_vertices = np.array(word_vertices, dtype=np.int32)
            word_contours.append(word_vertices)

            point = [int(np.mean(word_Xs)), int(np.mean(word_Ys))]
            word_points.append(point)

        # Also convert final to contour ndarray
        word_contours = np.array(word_contours, dtype=np.int32)

        self.ocr_contours = word_contours
        self.ocr_points = word_points

        #return word_contours, word_points

    def draw_panel_contours(self):
        cv2.drawContours(
            image = self.img_contoured,
            contours = self.panel_contours,
            contourIdx = -1,
            color = (255, 255, 0),
            thickness = 2
        )

    def draw_ocr_contours(self):
        cv2.drawContours(
            image = self.img_contoured,
            contours = self.ocr_contours,
            contourIdx = -1,
            color = (255, 0, 255),
            thickness = cv2.FILLED
        )

    def draw_ocr_points(self):
        for point in self.ocr_points:
            cv2.circle(self.img_contoured,
                       tuple(point),
                       radius=1,
                       color=(0, 255, 0),
                       thickness=2
            )

    def draw_textbox_outlines(self):
        cv2.drawContours(
            image = self.img_contoured,
            contours = self.textboxes,
            contourIdx = -1,
            color = (255, 255, 0),
            thickness = 2
        )


    # Check for word location within panel contours, add corresponding text
    def _assign_panel_text(self):
        """
        Group text according to which panel it appears in.

        Iterates through each panel contour, then checks each word's location
        for whether it's within the panel's bounds.

        """
        #TODO also check for overlap with textboxes
        # FIRST assign textboxes to panels, though
        self._text2coords()
        text_by_panel = []
        for panel_contour in self.panel_contours:
            textboxes = [
            ]
            current_panel_text = [
                text.description
                for text, ocr_point
                in zip(self.ocr_text[1:], self.ocr_points[1:])
                # Check if text is inside panel;
                # pointPolygonTest returns 1 if inside, -1 if outside, 0 if edge
                if
                cv2.pointPolygonTest(
                    contour = panel_contour,
                    pt = tuple(ocr_point),
                    measureDist=False
                ) > 0
                and 
                # This is where the text is filtered for matching with textboxes
                # TODO Probably not so efficient, though;
                # TODO First reduce set of textboxes to only those in current panel
                any(
                    cv2.pointPolygonTest(
                        contour = textbox,
                        pt = tuple(ocr_point),
                        measureDist=False
                    ) > 0
                    for textbox in self.textboxes
                )
            ]
            # Join separate list items into a single string and append
            text_by_panel.append(' '.join(current_panel_text))
        # get rid of empty strings for panels without text
        # (Why not keep it, though?)
        text_by_panel[:] = [x for x in text_by_panel if x != '']
        return text_by_panel


    def _saveContourImage(self):
        """
        Testing that drawContour successfully places both contour groups
        """
        # Display for testing purposes
        # cv2.imshow('circle', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        save_loc = (
            Path(self.SAVE_DEST_FOLDER) /
            f'{self.id:04d}_{self.filename.split(".")[0]}_contours'
            '.jpg'
        )
        cv2.imwrite(str(save_loc), self.img_contoured)
        # cv2.imshow('contours', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


    def parse(self):
        # TODO make a static variable?
        doc = Comic.NLP(' '.join(self.panel_text))
        self.sentences = [sent.constituency for sent in doc.sentences]


    def __repr__(self) -> str:
        """
        The text displayed when printing an instance of our sentence.
        """
        return f"""
    id:              {self.id}
    url:             {self.url}
    img_url:         {self.img_url}
    filename:        {self.filename}
    alt_text:        {self.alt_text}
    local_img_path:  {self.local_img_path}
    panel_text:      """ + "\n                     ".join(self.panel_text)
    

if __name__ == '__main__':

    # If json data already exists, load it
    if Path(JSONFILE).is_file():
        with open(JSONFILE, 'r') as read_file:
            comicsjson = json.load(read_file)
    else:
        comicsjson = {}

    comics = []

    # Iterate through range of comic ids
    for i in range(FIRST, LAST):
        comicdictkey = f'comic_{i}'

        if comicdictkey not in comicsjson:
            comic = Comic(i)
            comic.process_comic()
        # store key info
            comicsjson[comicdictkey] = {}
            comicsjson[comicdictkey]['alt_text'] = comic.alt_text
            comicsjson[comicdictkey]['comic_number'] = comic.id
            comicsjson[comicdictkey]['panel_text'] = comic.panel_text
            comicsjson[comicdictkey]['save_loc'] = str(comic.local_img_path)
        # show us what you got
            comic._saveContourImage()
            for line in comic.panel_text:
                if ('comeau' in line) or ('asofterworld' in line):
                    raise Exception(f'false positive in {comicdictkey}')
            comics.append(comic)
        else:
            print(f'{comicdictkey} already recorded. Skipping all.')


    with open('comics.json', 'w') as write_file:
        json.dump(comicsjson, write_file, sort_keys=True)


    end = timeit.default_timer()
    print(end - start)
    print((end - start)/60)
