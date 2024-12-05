"""
Created on Tue Oct 20 11:21:20 2020
@author: nelson-ryan
"""
# spaCy
# https://stackoverflow.com/questions/56896753/
# Stanza
# https://stanfordnlp.github.io/stanza/constituency.html

# TODO: Instead of filtering OCR results for those positioned within text
#       frames, filter OCR results for those positions within text boxes
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
    nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
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

        self.panel_frame_contours = []
        self.text_boxes = []
        self.ocr_text: Sequence = []
        self.ocr_contours: Sequence = []
        self.ocr_points: Sequence = []
        self.frame_text: list = []
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
        self.ocr_text = self.read_text()
        self.panel_frame_contours = self.find_panel_frames()
        self.show_img("frames")
        self.text_boxes = self.find_textboxes()
        self.show_img("text boxes")
        self.frame_text = self._assign_frame_text()
        self.show_img("text points")
        return


    def show_img(self, label) -> None:
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
                print("Saved" + str(self.local_img_path))


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


    def find_panel_frames(
        self,
        save_thresh: bool = False,
        save_blur = False
    ) -> np.ndarray:
        """
        Identify the boundaries of individual panels
        """
        # Load locally-saved image
        img = cv2.imread(str(self.local_img_path), cv2.IMREAD_UNCHANGED)

        # Threshold to get binary black-and-white
        _, framethresh = cv2.threshold(self.img_grey, 40, 255, cv2.THRESH_BINARY)

        # Check threshold result by saving image
        if save_thresh:
            cv2.imwrite(
                filename = (
                    f'{self.id:04d}_{self.filename.split(".")[0]}_thresh'
                    '.jpg'
                ),
                img = framethresh
            )
        # Median filter to remove jpg noise
        framethresh = cv2.medianBlur(framethresh, 3)
        if save_blur:
            cv2.imwrite(
                filename = (
                    f'{self.id:04d}_{self.filename.split(".")[0]}_blur'
                    '.jpg'
                ),
                img = framethresh
            )

        # Find contours
        contours, _ = cv2.findContours(
            framethresh,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE
        )

        contours = list(contours)
        contours.sort(
            key=lambda x: self._get_contour_precedence(x, img.shape[0])
        )

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

        cv2.drawContours(
            image = self.img_contoured,
            contours = contour_shortlist,
            contourIdx = -1,
            color = (255, 255, 0),
            thickness = 2
        )

        return contour_shortlist


    def find_textboxes(self):
        """I think the goal here is to better identify actual comic text
        versus anything in the photo
        This copies much from find_panel_frames
        Seems to work for comic #8
        """

        _, textboxthresh = cv2.threshold(self.img_grey, 230, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            textboxthresh,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Identify frames by including only contours with sufficient area
        contour_shortlist = []
        contour_shortlist = [
                cv2.boxPoints(cv2.minAreaRect(c)) for c in contours
                if 300 < cv2.contourArea(c) < 50000
        ]
        # Convert entire list to contour ndarray
        contour_shortlist = np.array(contour_shortlist, dtype=np.int32)

        cv2.drawContours(
            image = self.img_contoured,
            contours = contour_shortlist,
            contourIdx = -1,
            color = (255, 255, 0),
            thickness = 2
        )

        cv2.imwrite(f'comics/{self.id:04d}_textthresh.jpg',
                    textboxthresh)

        return contour_shortlist


    def read_text(self) -> Sequence:
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
        Reference: https://stackoverflow.com/questions/14161331/
        """
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

        self.ocr_contours = word_contours
        cv2.drawContours(
            image = self.img_contoured,
            contours = self.ocr_contours,
            contourIdx = -1,
            color = (255, 0, 255),
            thickness = 2
        )

        self.ocr_points = word_points
        for point in self.ocr_points:
            cv2.circle(self.img_contoured,
                       tuple(point),
                       radius=1,
                       color=(0, 255, 0),
                       thickness=2
            )

        #return word_contours, word_points


    # Check for word location within frame contours, add corresponding text
    def _assign_frame_text(self):
        """
        Group text according to which frame it appears in.

        Iterates through each panel contour, then checks each word's location
        for whether it's within the frame's bounds.

        """
        self._text2coords()
        text_by_frame = []
        for panel_frame_contour in self.panel_frame_contours:
            current_frame_text = []
            for text, ocr_point in zip(self.ocr_text[1:], self.ocr_points[1:]):
                # Check if text is inside frame;
                # pointPolygonTest returns 1 if yes
                if cv2.pointPolygonTest(
                    contour = panel_frame_contour,
                    pt = tuple(ocr_point),
                    measureDist=False
                ) > 0:
                    current_frame_text.append(text.description)
            # Join separate list items into a single string and append
            text_by_frame.append(' '.join(current_frame_text))
        # get rid of empty strings for frames without text
        text_by_frame = [x for x in text_by_frame if x != '']
        return text_by_frame


    def saveContourImage(self):
        """
        Testing that drawContour successfully places both contour groups
        """
        img = self.img.copy()
        # Display for testing purposes
        # cv2.imshow('circle', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        save_loc = (
            Path(self.SAVE_DEST_FOLDER) /
            f'{self.id:04d}_{self.filename.split(".")[0]}_contours'
            '.jpg'
        )
        cv2.imwrite(str(save_loc), img)
        # cv2.imshow('contours', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


    def parse(self):
        # TODO make a static variable?
        doc = Comic.nlp(' '.join(self.frame_text))
        self.sentences = [x.constituency for x in doc.sentences]
        # for sentence in doc.sentences:
            # print(sentence.constituency)


    def __repr__(self) -> str:
        """
        The text displayed when printing an instance of our sentence.
        """
        return f"""
        self.id:                   {self.id}
        self.url:                  {self.url}
        self.soup:                 {len(self.soup)}
        self.img_url:              {self.img_url}
        self.alt_text:             {self.alt_text}
        self.filename:             {self.filename}
        self.local_img_path:       {self.local_img_path}
        self.img:                  {len(self.img)}
        self.panel_frame_contours: {len(self.panel_frame_contours)}
        self.text_boxes:           {len(self.text_boxes)}
        self.ocr_text:             {len(self.ocr_text)}
        self.ocr_contours:         {len(self.ocr_contours)}
        self.ocr_points:           {len(self.ocr_points)}
        self.frame_text:           {len(self.frame_text)}
        self.sentences:            {(self.sentences)}
        """
    

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
            comicsjson[comicdictkey]['frame_text'] = comic.frame_text
            comicsjson[comicdictkey]['save_loc'] = str(comic.local_img_path)
        # show us what you got
            comic.saveContourImage()
            for line in comic.frame_text:
                if ('comeau' in line) or ('asofterworld' in line):
                    raise Exception(f'false positive in {comicdictkey}')
            comics.append(comic)
        else:
            print(f'{comicdictkey} already recorded. Skipping all.')


    with open('comics.json', 'w') as write_file:
        json.dump(comicsjson, write_file, sort_keys=True)
