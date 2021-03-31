# -*- coding: utf-8 -*-
"""
Leyla Tarhan
lytarhan@gmail.com
3/2021

utility functions for exploring and explaining NLP features

"""

# %% libraries

from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

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
    
# %% print distance matrices to assess inter-subject reliability of the feature embeddings

# - for each video, compare the feature embeddings extracted from individual 
# subjects' descriptions of that video. 
# - do this by computing the euclidean distance between embedding 
# vectors -- captures differences in overall magnitude of features as well as 
# relative patterns across features
# - result = a square (subs x subs) distance matrix, measuring how similar the embeddings are across subjects.   
# - done separately for each video set


def printInterSubReliability(featureDF):
    # calculate inter-sub reliability for each video set
    stimSets = featureDF['stimSet'].unique();
    
    fig, axes = plt.subplots(1, len(stimSets), figsize=(15, 6))
    fig.suptitle('comparing embeddings across subjects')
    
    for s in stimSets:
        currSetDF = featureDF[featureDF['stimSet'] == s];
        
        # set up a distance matrix to keep track of sub reliability across videos
        nSubs = len(currSetDF['subName'].unique())
        dists = np.zeros((nSubs, nSubs))
        
        # loop through the vids in this stim set
        currSetVids = currSetDF['vidName'].unique();
        
        for v in currSetVids:
            # get feature embeddings for this vid's description, from all subs
            currVidFeatures = currSetDF[currSetDF['vidName'] == v];
            currVidFeatures = currVidFeatures.set_index('subName'); # keep track of the subs
            currVidFeatures = currVidFeatures.sort_index(); # sort by sub to maintain the same order every time
            currVidFeatures = currVidFeatures.drop(columns = ['vidName', 'stimSet']) # get just the embeddings
            
            x = euclidean_distances(currVidFeatures, currVidFeatures);
            # result = square matrix of e.d.'s (dims = subs x subs) (numpy array)
            
            # get absolute value of these distances (so they don't just cancel out when we average) and add to the distance matrix:
            dists = dists + np.absolute(x);
            
            
        # complete the averaging process: divide distances by # vids
        nVids = len(currSetVids)
        dists = dists / nVids
        
        # figure out which subplot we're in
        sp = np.where(stimSets == s)[0][0]
        
        # print out a heatmap of the distance matrix
        mask = np.zeros_like(dists)
        mask[np.triu_indices_from(mask)] = True          
        sns.heatmap(dists, mask = mask, vmin = 0, square = True, 
                    ax = axes[sp], 
                    xticklabels = currSetDF['subName'].unique(), 
                    yticklabels = False, 
                    cbar_kws={'label': 'avg. euclidean distance', 'orientation': 'vertical'})
        axes[sp].set_title('Video set: ' + s)
                

# %% Exclude outlier subjects

def excludeOutliers(df, subsToExclude):
    print('hello!')
        
        
  