# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 11:21:20 2020
@author: nelsonr

This section of code defines a function to download and save
all A Softer World comics from the website, utilizing the
Beautiful Soup package.

"""

import bs4  # beautifulsoup4
import requests
import os  # for directory checking and creation
import cv2

import comicsplit2
import vision_ocr


# UPDATE FIRST COMIC NUMBER IN VARIABLE DECLARATION
def main():
    comics = []
    pulling_comic = 1247
    while True:
        retrieved_comic = save_comic(pulling_comic)
        if retrieved_comic:
            comics.append(retrieved_comic)
            pulling_comic += 1
        else:
            break

    # Use same destination path stored in dict by save_comic
    for i in range(len(comics)):
        comic_path = comics[i].get("save_loc")
        print(comic_path)
        frame_contours = comicsplit2.find_frames(comic_path)
        ocr_text = vision_ocr.detect_text(comic_path)
        ocr_contours, ocr_points = vision_ocr.text2coords(ocr_text)
        '''All of these ocr_* are convoluted; this might be a ideal place to use a class'''
        # Testing that drawContour successfully places both contour groups
        # img = cv2.imread(comic_path, cv2.IMREAD_UNCHANGED)
        # cv2.drawContours(img, frame_contours, -1, (255, 255, 0), 2)
        # cv2.drawContours(img, ocr_contours, -1, (255, 0, 255), 2)
        # for point in ocr_points:
        #     cv2.circle(img, tuple(point), radius=3, color=(0, 255, 0), thickness=3)
        # cv2.imshow('circle', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # TODO Check for overlap of each OCR contour against the
        #  frame contours and add the corresponding text accordingly
        #  Use cv2.pointPolygonTest to achieve this:
        #  https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_contours/py_contours_more_functions/py_contours_more_functions.html
        text_by_frame = []
        for j in range(len(frame_contours)):
            text_by_frame.append([])
            for k in range(1, len(ocr_points)):
                # Check if text is inside frame; pointPolygonTest returns 1 if yes (-1 if no, 0 if eq)
                if cv2.pointPolygonTest(frame_contours[j],
                                        tuple(ocr_points[k]),
                                        False) > 0:
                    text_by_frame[j].append(ocr_text[k].description)
        print(text_by_frame)


# Get individual comic info and save it to a dictionary
def save_comic(n):
    comic_dict = {}
    comic_number = n
    res = requests.get(f'https://www.asofterworld.com/index.php?id={comic_number}')
    res.raise_for_status()
    
    soup = bs4.BeautifulSoup(res.text, "html.parser")
    comic = soup.select("#comicimg > img")
    
    url = comic[0].get('src')

    filename = url.split('/')[-1] # filename is last part of URL, following the last slash
    # If there's no filename, there's no comic, so stop
    if not filename:
        return
    alt_text = comic[0].get('title')

    img_url = requests.get(url)

    if not os.path.exists('comics/'):
        os.mkdir('comics/')
    save_loc = f'comics/{comic_number:04d}_{filename}'
    with open(save_loc, 'wb') as img:
        img.write(img_url.content)

    # These could well also be class attributes
    comic_dict["comic_number"] = comic_number
    comic_dict["filename"] = filename
    comic_dict["alt_text"] = alt_text
    comic_dict["save_loc"] = save_loc

    return comic_dict


def group_frame_text(frame, text):
    return


main()
