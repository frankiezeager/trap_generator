#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 15:32:24 2017

@author: frankiezeager

Lyric Scraper
Input: Artist
Output: full list of lyrics from that artist from AZ Lyrics
"""

import re
import requests
from bs4 import BeautifulSoup
import time

def get_lyrics(artist):
    t0 = time.time()
    headers = {    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.10; rv:30.0) " + 
                  "Gecko/20100101 Firefox/30.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive"}
    #clean artist name to match AZ Lyrics url
    artist = re.sub('[^A-Za-z0-9]+', "", artist)
    if artist.startswith("the"):
        artist = artist[3:]
    
    #collect and parse artist page
    base_url="https://www.azlyrics.com/m/"+artist+".html"
    page=requests.get(base_url,headers=headers)
    soup=BeautifulSoup(page.text,'html.parser')
    
    #pull all text from the album div
    all_songs=soup.find('div',id='listAlbum')
    song_list=all_songs.find_all(href=True)
    
    #pull link for each song
    #then grab lyrics from the links
    lyrics_list=[]
    #set up the partitions of where the lyrics lie on the page
    beginning_marker = '<!-- Usage of azlyrics.com content by any third-party lyrics provider is prohibited by our licensing agreement. Sorry about that. -->'
    end_marker = '<!-- MxM banner -->'
    for song in song_list:
        link=song.attrs['href']
        if link.startswith('..'):
            link=link.replace('..','http://www.azlyrics.com')
        
        #get the lyrics text for each link
        lyric_page=requests.get(link,headers=headers)
        soup=BeautifulSoup(lyric_page.text,'html.parser')
        lyrics=lyric_page.split(beginning_marker)[1]
        lyrics=lyrics.split(end_marker)[0]
        lyrics_list.append(lyrics)
        
    #intentional lag to not overwhelm the servers, proportional to how long it took the server to respond
    response_delay = time.time() - t0
    time.sleep(10*response_delay)
    return lyrics_list
 
get_lyrics('migos')