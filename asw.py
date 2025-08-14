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
from stanza import Pipeline
from functools import cached_property

FIRST = 1
LAST = 1249 # (1248 comics, non-inclusive range)
JSONFILE = 'comics.json'

# Google Vision credential
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
    "nomadic-zoo-293819-eecd00e68b8e.json"
)

class Comic:
    """ASofterWorld individual comic object."""

    NLP = Pipeline(
        lang='en',
        processors='mwt,tokenize,pos,constituency',
        logging_level="WARN"
    )
    BASE_URL = 'https://www.asofterworld.com/index.php?id='

    IMG_FOLDER = 'comics'
    SAVE_DEST_FOLDER = Path(IMG_FOLDER)
    CLIENT = vision.ImageAnnotatorClient()


    def __init__(self, number: int):
        self.id: int = number
        self.ocr_points = [] # Should be revised; this is set by ocr_contours


    @property
    def url(self):
        """The URL to an individual comic's full webpage.
        Scraped by the .soup() method."""
        return self.BASE_URL + str(self.id)


    @property
    def hover(self):
        """The image's 'title' text."""
        return str(self.soup.get('title'))


    @property
    def filename(self):
        """The filename as retrieved from asofterworld"""
        return self.img_url.split('/')[-1]


    @property
    def local_img_path(self):
        """The filepath where a local copy of the image file is saved"""
        return Path(self.SAVE_DEST_FOLDER) / f'{self.id:04d}_{self.filename}'


    @cached_property
    def img(self):
        """The comic image.
        When accessing this property for the first time, this downloads and
        saves the image file unless the local file is already present."""
        self._download_jpg()
        img = cv2.imread(str(self.local_img_path))
        return img


    @cached_property
    def img_grey(self):
        """Greyscale image copy,
        used for finding textbox rectangles and panel boundaries."""
        if len(self.img.shape) == 2:
            return self.img.copy()
        else:
            return cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)


    @cached_property
    def img_contoured(self):
        """Image copy for drawing onto"""
        return self.img.copy()


    def show_img(self) -> None:
        """Display whichever image in a temporary window."""
        cv2.imshow(self.hover,
                   self.img_contoured)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return None


    @cached_property
    def soup(self) -> bs4.element.Tag:
        """
        Retrieves and extracts the image information from asofterworld's html.
        """
        print(f"Retrieving html from {self.url}")
        res = requests.get(self.url)
        res.raise_for_status()
        soup_element_tag = (
            bs4.BeautifulSoup(res.text, "html.parser")
            .select("#comicimg > img")
        )[0]
        return soup_element_tag


    @cached_property
    def img_url(self):
        try:
            return str(self.soup.get('src'))
        except IndexError:
            raise Exception(f'Comic {self.id} does not exist')


    def _download_jpg(self):
        """Downloads the comic jpg locally, if the image isn't already saved.
        Includes correction of file-final image data,
        which a few of the online comic images are missing."""
        if self.local_img_path.exists():
            print(f'{self.local_img_path} already exists. Skipping download.')
        else:
            print(f"Downloading f{self.filename} to {self.local_img_path}")
            if not self.SAVE_DEST_FOLDER.exists():
                self.SAVE_DEST_FOLDER.mkdir()
            with open(self.local_img_path, 'wb') as img:
                img_url = requests.get(self.img_url)
                img.write(img_url.content)
            self._fix_broken_jpg()
            print("Saved " + str(self.local_img_path))


    def _fix_broken_jpg(self):
        """Correct missing file-final image data, if applicable.
        Such is the case, for example, with #363.
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
        Used for sorting panel frames sequentially, especially those that
        don't fall neatly in the Cartesian plane when there are more than three
        panels.
        This function comes from https://stackoverflow.com/a/39445901
        """
        tolerance_factor = 50
        origin = cv2.boundingRect(contour)
        return (origin[1] // tolerance_factor) * cols + origin[0]


    @cached_property
    def panel_contours(
        self,
        save_thresh: bool = False,
        save_blur = False
    ) -> np.ndarray:
        """Identify the boundaries of individual panels."""

        # Threshold to get binary black-and-white
        # This uses a different threshold than textboxes does because these have
        # starker contrast
        _, self.framethresh1 = cv2.threshold(
            src = self.img_grey,
            thresh = 40,
            maxval = 255,
            type = cv2.THRESH_BINARY
        )

        # Median filter to remove jpg noise
        self.frameblur = cv2.medianBlur(self.framethresh1, 3)

        # Find contours
        contours, _ = cv2.findContours(
            image = self.frameblur,
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


    @cached_property
    def textboxes(self):
        """The regions of comic text based on the white label-maker rectangles.
        Each region is assigned to a panel based on where its center lies.
        List of list of contours.
        """

        # Threshold to get binary black-and-white
        # This uses a different threshold than panel_contours does because these
        # may have subtler contrast versus the photo colors
        _, self.img_textboxthresh = cv2.threshold(
            src = self.img_grey,
            thresh = 240,
            maxval = 255,
            type = cv2.THRESH_BINARY
        )
        contours, _ = cv2.findContours(
            image = self.img_textboxthresh,
            mode = cv2.RETR_LIST,
            method = cv2.CHAIN_APPROX_SIMPLE
        )

        # Did removing this improve reliability?
        # Median filter to remove jpg noise
        # self.frameblur = cv2.medianBlur(self.framethresh1, 3)

        # List of textbox lists, assigned according to which panel they're in
        textboxes_by_panel = [ [] for _ in self.panel_contours ]
        for c in contours:
            boxpoints = cv2.boxPoints(cv2.minAreaRect(c))
            # filter by size
            if 300 < cv2.contourArea(boxpoints) < self.smallestpanel: # < 20000:
                boxcenter = (
                    int(np.mean([x[0] for x in boxpoints])),
                    int(np.mean([x[1] for x in boxpoints]))
                )

                for panel_idx, panel in enumerate(self.panel_contours):
                    if cv2.pointPolygonTest(
                            contour = panel,
                            pt = boxcenter,
                            measureDist=False
                    ) > 0:
                        textboxes_by_panel[panel_idx].append(
                            np.array(boxpoints, dtype = np.int32)
                        )

        return textboxes_by_panel


    @cached_property
    def ocr_text(self):
        """
        Use Google Vision OCR to read text in the image.
        """

        with io.open(self.local_img_path, 'rb') as image_file:
            content = image_file.read()
        image = vision.Image(content = content)

        print("Calling Google Vision OCR")
        response = Comic.CLIENT.text_detection(image = image)
        texts = response.text_annotations

        # Keeping this print line here to reference the description object
        #print('\n"{}"'.format(texts[0].description))

        if response.error.message:
            raise Exception(
                '{}\nFor more info on error messages, check: '
                'https://cloud.google.com/apis/design/errors'.format(
                    response.error.message))

        return texts


    @cached_property
    def ocr_contours(self):
        """Create cv2 contour list from texts coordinates,
        Defines values for self.ocr_contours and self.ocr_points

        Reference: https://stackoverflow.com/questions/14161331/
        """
        word_contours = []  # for storing all words
        word_points = []
        new_text_tuples = []

        for text in self.ocr_text[1:]: # skip the first, with all text
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

        self.ocr_points = word_points
        return word_contours


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
        for t in self.textboxes:
            cv2.drawContours(
                image = self.img_contoured,
                contours = t,
                contourIdx = -1,
                color = (255, 255, 0),
                thickness = 2
            )


    def draw_ocr_text(self):
        """
        Recreates the labelmaker text and overlays it onto the image.
        """
        for panel in self.panel_text:
            for text in panel:
                contours = np.array( [
                    [ [vertex.x, vertex.y]
                         for vertex
                         in text.bounding_poly.vertices ]
                    ], dtype = np.int32
                )
                cv2.drawContours(
                    image = self.img_contoured,
                    contours = contours,
                    contourIdx = -1,
                    color = (255, 200, 200),
                    thickness = cv2.FILLED
                )
                cv2.putText(
                    self.img_contoured,
                    text = text.description,
                    org = tuple(contours[0][3]),
                    fontFace = cv2.FONT_HERSHEY_COMPLEX,
                    fontScale = .38,
                    color = (120, 20, 80)
                )


    @cached_property
    def panel_text(self) -> list:

        assigned_text = [ [] for _ in self.panel_contours ]
        for ocr_text in self.ocr_text[1:]:
            vertices = ocr_text.bounding_poly.vertices
            centerpoint = (
                int(np.mean([vertex.x for vertex in vertices])),
                int(np.mean([vertex.y for vertex in vertices]))
            )
            # cv2.circle(
            #     img = self.img_contoured,
            #     center = centerpoint, radius = 3, color = (255,0,255), thickness = cv2.FILLED
            # )
            for panel_i, panel_textboxes in enumerate(self.textboxes):
                # Check if text is inside panel;
                # pointPolygonTest returns 1 if inside, -1 if outside, 0 if edge
                if any(
                        cv2.pointPolygonTest(
                            contour = textbox,
                            pt = centerpoint,
                            measureDist = False
                        ) > 0
                        for textbox in panel_textboxes
                ):
                    assigned_text[panel_i].append(ocr_text)
        return assigned_text


    def _saveContourImage(self):
        """
        For visually confirming whether drawContour successfully places both contour groups
        """
        save_loc = (
            Path(self.SAVE_DEST_FOLDER) /
            f'{self.id:04d}_{self.filename.split(".")[0]}_contours'
            '.jpg'
        )
        cv2.imwrite(str(save_loc), self.img_contoured)


    @property
    def textbypanel(self):
        """Joined text for each panel."""
        return [
            " ".join([t.description for t in panel])
            for panel in self.panel_text
        ]


    @cached_property
    def smallestpanel(self):
        """The smallest panel contour, used in filtering .textboxes"""
        return min([cv2.contourArea(panel) for panel in self.panel_contours])


    @cached_property
    def parse(self):
        # TODO make a static variable?
        doc = Comic.NLP(' '.join(self.textbypanel))
        return [sent.constituency for sent in doc.sentences]


    def __repr__(self) -> str:
        """
        The text displayed when printing an instance of our sentence.
        """
        return f"""
    id:              {self.id}
    url:             {self.url}
    img_url:         {self.img_url}
    filename:        {self.filename}
    hover:           {self.hover}
    local_img_path:  {self.local_img_path}
    panel_text:      """ + "\n                     ".join(self.textbypanel)
    

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
        # store key info
            comicdict = {}
            comicdict['hover']     = comic.hover
            comicdict['comic_number'] = comic.id
            comicdict['panel_text']   = comic.panel_text
            comicdict['save_loc']     = str(comic.local_img_path)
            comicsjson[comicdictkey]  = comicdict
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

