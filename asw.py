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


# UPDATE FIRST COMIC NUMBER IN FUNCTION CALL
def main():
    comics = []
    currentcomic = 1245
    while True:
        gotcomic = save_comic(currentcomic)
        if gotcomic:
            comics.append(gotcomic)
            currentcomic += 1
        else:
            break
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

    filename = url.split('/')[-1] # filename is last part of URL, following the last slash
    # If there's no filename, there's no comic, so stop
    if not filename:
        return

    alttext = comic[0].get('title')

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

