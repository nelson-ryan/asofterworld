# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 11:21:20 2020
@author: xople

This section of code defines a function to download and save
all A Softer World comics from the website, utilizing the
Beautiful Soup package.

"""

import bs4 #beautifulsoup4
import requests

# UPDATE COMIC RANGE IN FUNCTION CALL
def main():
    for i in range(700,705):
        save_comic(i)


def save_comic(n):
    
    comicnumber = n
    res = requests.get(f'https://www.asofterworld.com/index.php?id={comicnumber}')
    res.raise_for_status()
    
    soup = bs4.BeautifulSoup(res.text, "html.parser")
    
    comic = soup.select("#comicimg > img")
    
    url = comic[0].get('src')
    alttext = comic[0].get('title')
    filename = url.split('/')[-1]
    # print(comicnumber)
    # print(alttext)
    # print(filename)
    
    imgurl = requests.get(url)
    res.raise_for_status()
    
    with open(f'comics/{comicnumber:04d}_{filename}', 'wb') as img:
        img.write(imgurl.content)

main()

