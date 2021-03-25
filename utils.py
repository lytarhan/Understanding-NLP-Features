# -*- coding: utf-8 -*-
"""
Leyla Tarhan
lytarhan@gmail.com
3/2021

utility functions for exploring and explaining NLP features

"""

# %% print out descriptions for any video

# 1. prints out a list of the vid names
# 2. asks you to enter a vid name (or quit)
# 3. prints out all the descriptions of that vid
# 4. asks you to do it again (or quit)

def printDescriptions(df):
    # 1. print out a list of the vids
    uniqueVids = sorted(df['vidName'].unique()) # get unique vid names, and sort alphabetically
    print('Action videos we have data for: \n')
    print(*uniqueVids, sep = "\n")
    
    # 2. ask for input
    vid = input('Enter video name to see how people described it, or press q to quit: ') # type = string
    
    done = 0
    while not done:
        
        if vid == 'q': # hit the quit key
            done = 1;
            print('...quitting the description-explorer.')
            break
        
        # make sure the vid name is in our list
        uniqueVidsLower = [i.lower() for i in uniqueVids]
        if vid.lower() in uniqueVidsLower:
            # return to the specific case used in the dataframe of descriptions
            caseSensitiveVid = uniqueVids[uniqueVidsLower.index(vid.lower())]
            
            # 3. get descriptions for the video they specified
            currVidDescr = df[df['vidName'] == caseSensitiveVid]
            
            print('Descriptions for ' + caseSensitiveVid + ':')
            print('------------------------------')
            print(*currVidDescr['description'], sep = '\n')
            
            # 4. check if they want to ask about another vid
            vid = input('Enter another video name, or press q to quit: ')
            
        else: # they entered a bad vid name
            print('sorry, no data for ' + vid + '.')
            vid = input('Enter another video name, or press q to quit: ')
    
     