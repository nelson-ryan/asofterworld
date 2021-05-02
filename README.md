# A Softer World Text Analysis

This is the start of a project that I first thought of on May 27, 2015, just a few days before the final A Softer World comic.

The spirit of the idea is to look for any sort of pattern in how sentences are broken up between panels in A Softer World comics, relative to the structure of the sentences. Because I had often found the breaks to be counter-intuitive, I might expect a high occurence of the breaks being within major syntactic constituents/phrases, rather than at their boundaries.

Identifying information for each comic (title, number, ~~date,~~ *date does not appear to be available* alt-text), as well as the comics themselves, will then be collected from the website. Each comic will be separated into individual frames, from which OCR will extract the text. Likely using the NLTK, sentences will be notated with syntactic breaks. Then, combining all of these pieces together (I'm looking into whether JSON would be a useful/appropriate structure for this), I will then analyze to look for any patterns. 

asw.py downloads all ASW comics from the website, using Beautiful Soup. Title and filename are currently used and alt-text extracted.

comicsplit2.py separates each comic's frames into separate files, and now does so in the correct order. This currently saves each frame to its own file, however the future plan is to match the bounding vertices from the OCR output to the bounding boxes that this function identifies within the comic.

vision_ocr.py uses Google Cloud Vision (the API key is not included in the repo, of course) to read text from the images. The plan is for this to return the bounding vertices to be compared to those identified in comicsplit2.

