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
#now we have the ~2100 songs lyrics in one string

#remove the <br />and <div/> tags (but keep the new line characters to maintain format)
clean_lyrics=re.sub(r'<[^<]+?>', '', flattened_lyrics)

#remove things like [Chorus:]
clean_lyrics=re.sub(r'\[.+\]', '', clean_lyrics)

#remove encoding characters like \x86
clean_lyrics=re.sub(r'[^\x00-\x7f]','',clean_lyrics)
clean_lyrics=re.sub(r'\#','',clean_lyrics)
#remove punctuation
clean_lyrics=re.sub(r'[^A-Za-z0-9\s]','',clean_lyrics)




#convert to lowercase
clean_lyrics=clean_lyrics.lower()
clean_lyrics[0:1000]
'?' in clean_lyrics.split()
#chars = sorted(list(set(clean_lyrics)))
#list(set(re.sub('\\[x].{2}','',clean_lyrics)))
pickle.dump(clean_lyrics,open('clean_lyrics.p','wb'))
clean_list_lyrics = []
### create list of lyrics as one giant string
for artist in lyrics:
    for i in artist:
        #remove the <br />and <div/> tags (but keep the new line characters to maintain format)
        lyric=re.sub(r'<[^<]+?>', '', i)

        #remove things like [Chorus:]
        lyric=re.sub(r'\[.+\]', '', lyric)

        #remove encoding characters like \x86
        lyric=re.sub(r'[^\x00-\x7f]','',lyric)

        #remove punctuation
        lyric=re.sub(r'\?|\!|\,|\.|\'|\"|\(|\)|\-','',lyric)


        #convert to lowercase
        lyric=lyric.lower()
        clean_list_lyrics.append(lyric)

clean_list_lyrics[0:10]
pickle.dump(clean_list_lyrics,open('clean_list_lyrics.p','wb'))
