#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 16:41:18 2017

@author: frankiezeager

Clean Lyrics
"""
import pickle
#from html.parser import HTMLParser
import re

lyrics=pickle.load(open('trap_lyrics.p','rb'))


#unnest the list of lyrics
flattened_lyrics=[]
for sublist in lyrics:
    for item in sublist:
        flattened_lyrics.append(item)
flattened_lyrics=''.join(flattened_lyrics)
#now we have the ~2100 songs lyrics in a list from the 11 artists

#remove the <br />and <div/> tags (but keep the new line characters to maintain format)
clean_lyrics=re.sub('<[^<]+?>', '', flattened_lyrics)

#remove characters \xa0, \x82,\x99,\x95,\x93,\x94,'\x9c','\x83','\x80',\x8b',\xad,\x98,\x9d'
#list(set(re.sub('\\[x].{2}','',clean_lyrics)))
pickle.dump(clean_lyrics,open('clean_lyrics.p','wb'))