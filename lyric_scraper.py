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
import pickle

def get_random_profile():
    """
    Gets list of potential user agents.
    Args: 
        None
    Returns:
        string: a random user agent to mimic browser access
    """
    
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
    """Function to access AZ Lyrics with random user profile and gather all song lyrics for an artist
    Args:
        artist(string): The artist you would like to gather all songs for
    Returns:
        list: a list of strings of the lyrics for each song
    """
    
    #clean artist name to match AZ Lyrics url
    artist = re.sub('[^A-Za-z0-9]+', "", artist)
    artist=artist.lower()
    if artist.startswith("the"):
        artist = artist[3:]
    #initialize song list   
    song_list=[]
    try:
        #collect and parse artist page
        if artist[0].isdigit():
            first_letter='19'
        else:
            first_letter=artist[0]
        base_url="https://www.azlyrics.com/"+first_letter+"/"+artist+".html"
        headers=get_random_profile()
        page=requests.get(base_url,headers=headers)
        soup=BeautifulSoup(page.text,'html.parser')
        
        #pull all text from the album div
        all_songs=soup.find('div',id='listAlbum')
        song_list=all_songs.find_all(href=True)
        
        #pull link for each song
        #then grab lyrics from the links
        
        #set up the partitions of where the lyrics lie on the page
        beginning_marker = '<!-- Usage of azlyrics.com content by any third-party lyrics provider is prohibited by our licensing agreement. Sorry about that. -->'
        end_marker = '<!-- MxM banner -->'   
    
        i=1
        lyric_list=[]
        for song in song_list:        
            #insert user agent in headers to mimic browser user
            headers=get_random_profile()
            
            #get link for each song
            link=song.attrs['href']
            
            if link.startswith('..'):
                link=link.replace('..','http://www.azlyrics.com')
            
            try:      
            #get the lyrics text for each link
                lyric_page=requests.get(link,headers=headers)
                soup=BeautifulSoup(lyric_page.text,'html.parser')
                soup=str(soup)        
                lyrics=soup.split(beginning_marker)[1]
                lyrics=lyrics.split(end_marker)[0]
                lyric_list.append(lyrics)
                
                print("imported song number ",i,"of ",len(song_list),"for artist",artist)
                i=i+1
            
            except:
                
                print("error for song ",i,"of ",len(song_list),"for artist ",artist)
            
            #intentional random delay to not overwhelm the servers
            time.sleep(random.randint(10,20))
    
    except :
        print("Error for artist ", artist)
        time.sleep(random.randint(15,30))
        
    time.sleep(random.randint(15,30))
    
    return lyric_list


#run with specified trap artists
trap_artists=['2 Chainz','Gucci Mane','Waka Flocka Flame','Young Thug',
              'Fetty Wap','Lil Uzi Vert','Lil Yachty','21 Savage','Young Jeezy','migos','Rae Sremmurd']

all_lyrics=[]
for artist in trap_artists:
    lyric=get_lyrics(artist)
    all_lyrics.append(lyric)
    
pickle.dump(all_lyrics,open('trap_lyrics.p','wb'))