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

import comicsplit2
#import vision_ocr #

# UPDATE FIRST COMIC NUMBER IN VARIABLE DECLARATION
def main():
    comics = []
    current_comic = 1245
    while True:
        got_comic = save_comic(current_comic)
        if got_comic:
            comics.append(got_comic)
            current_comic += 1
        else:
            break

    # Use same destination path stored in dict by save_comic
    for i in range(len(comics)):
        comicsplit2.split_frames(comics[i].get("save_loc"))


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

    comic_dict["comic_number"] = comic_number
    comic_dict["filename"] = filename
    comic_dict["alt_text"] = alt_text
    comic_dict["save_loc"] = save_loc

    return comic_dict


main()
