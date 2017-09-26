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
import random

import socks
import socket

import stem.process


def get_random_profile():
    #list of potential user agents
    user_agents = [
    'Mozilla/5.0 (Windows; U; Windows NT 5.1; it; rv:1.8.1.11) Gecko/20071127 Firefox/2.0.0.11',
    'Opera/9.25 (Windows NT 5.1; U; en)',
    'Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)',
    'Mozilla/5.0 (compatible; Konqueror/3.5; Linux) KHTML/3.5.5 (like Gecko) (Kubuntu)',
    'Mozilla/5.0 (Windows NT 5.1) AppleWebKit/535.19 (KHTML, like Gecko) Chrome/18.0.1025.142 Safari/535.19',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.7; rv:11.0) Gecko/20100101 Firefox/11.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.6; rv:8.0.1) Gecko/20100101 Firefox/8.0.1',
    'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.19 (KHTML, like Gecko) Chrome/18.0.1025.151 Safari/535.19']
    
    #get random user agent
    user_agent=random.choice(user_agents)
    
    #put in header
    headers = {"User-Agent": user_agent,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive"}
    
    return headers




def get_lyrics(artist):
    #clean artist name to match AZ Lyrics url
    artist = re.sub('[^A-Za-z0-9]+', "", artist)
    if artist.startswith("the"):
        artist = artist[3:]
    
    #collect and parse artist page
    base_url="https://www.azlyrics.com/m/"+artist+".html"
    headers=get_random_profile()
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
        #insert user agent in headers to mimic browser user
        headers=get_random_profile()
        
        #get link for each song
        link=song.attrs['href']
        
        if link.startswith('..'):
            link=link.replace('..','http://www.azlyrics.com')
        
        #get the lyrics text for each link
        lyric_page=requests.get(link,headers=headers)
        soup=BeautifulSoup(lyric_page.text,'html.parser')
        soup=str(soup)        
        lyrics=soup.split(beginning_marker)[1]
        lyrics=lyrics.split(end_marker)[0]
        lyrics_list.append(lyrics)
        print(lyrics)
        
        #intentional random delay to not overwhelm the servers
        time.sleep(random.randint(10,20))
    time.sleep(random.randint(15,30))
    
    return lyrics_list

 
migos_lyrics=get_lyrics('migos')