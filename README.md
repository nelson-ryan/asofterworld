# A Softer World Text Analysis

This is the start of a project that I first thought of on May 27, 2015, just a few days before the final A Softer World comic.

The spirit of the idea is to look for any sort of pattern in how sentences are
broken up between panels in A Softer World comics, relative to the structure
of the sentences. Because I had often found the breaks to be counter-intuitive,
I might expect a high occurence of the breaks being within major syntactic
constituents/phrases, rather than at their boundaries.

asw.py downloads all A Softer World comics from the website, using Beautiful Soup. I still want to modify this to extract the alt-text (which is 'title') and match it with each comic's text and filename. This could probably go into an SQL database, where the OCR output might be added, as well, and potentially categorize them there by whatever type of syntactic breaks.

vision_ocr.py uses a Google Vision API to read text from the comics.

comicsplit2.py separates each comic's frames into separate files. The output is not in the correct order, however, so more work is needed.

Once the ordering issue with the comic panels is fixed, the panels can be read by the OCR in order.

Next steps will be to use Natural Language Toolkit and perhaps statistical packages to look for patterns, if any, in where the breaks between panels occur.

