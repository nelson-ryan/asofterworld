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


# UPDATE COMIC RANGE IN FUNCTION CALL
def main():
    comics = []
    for i in range(700, 705):
        comics.append(save_comic(i))
    for i in range(len(comics)):
        print(comics[i])


def save_comic(n):
    comicdict = {}
    comicnumber = n
    res = requests.get(f'https://www.asofterworld.com/index.php?id={comicnumber}')
    res.raise_for_status()
    
    soup = bs4.BeautifulSoup(res.text, "html.parser")
    comic = soup.select("#comicimg > img")
    
    url = comic[0].get('src')
    alttext = comic[0].get('title')
    filename = url.split('/')[-1] # filename is last part of URL, following the last slash

    comicdict["comicnumber"] = comicnumber
    comicdict["filename"] = filename
    comicdict["alttext"] = alttext

    if not os.path.exists('comics/'):
        os.mkdir('comics/')

    imgurl = requests.get(url)
    with open(f'comics/{comicnumber:04d}_{filename}', 'wb') as img:
        img.write(imgurl.content)

    return comicdict

main()

