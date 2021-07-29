# A Softer World Text Analysis

A Softer World was a sort of webcomic that ran from 2003 to 2015. Its format was similar to three-panel comics, with short text superimposed over still photographs.

The goal of the project is to look for any sort of pattern in how sentences are broken up between panels in A Softer World comics, relative to the structure of the sentences. Because I had often found the breaks to be counter-intuitive, I might expect a high occurence of the breaks being within major syntactic constituents/phrases, rather than at their boundaries.

Information from each comic--title, number, alt-text--is stored in a Python dictionary. Google Vision OCR extracts the text, which is also stored in the dictionary based on the comic frame in which it appears. The future plan is to make use of the NLTK (research needed to determine how this might be done) to notate the text with syntactic constituent boundaries and compare those to breaks between comic frames.
