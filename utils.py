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
    print('\nCalculating how similar feature embeddings were across subs...')
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
        
    print('check your plots to see how similar feature embeddings were across subjects.\n')
                

# %% Exclude outlier subjects

def excludeOutliers(df, subsToExclude):
    print('\nExcluding outlier subjects...')
    if len(subsToExclude) > 0:
        # assign sub name as index for this data frame
        df = df.set_index('subName')
        
        # drop all data for the excluded subs
        cleanedDF = df.drop(subsToExclude)
        
        # double-check that this worked
        x = [sub in cleanedDF.index.unique() for sub in subsToExclude]
        assert not any(x), "double-check that all excluded subs were dropped."
        
        print(str(len(subsToExclude)) + ' outlier subs were dropped.\n')
        
        return cleanedDF
    else:
        print('0 subs were dropped.\n')

    
# %% Compare split-half reliability within vs. between video exemplars

def withinBetweenReliability(df):   
    print('\nComparing NLP embeddings within and across exemplars...')
    
    nIters = 10; # how many split-halves to run in the reliability calculation?
    
    # set up a dataframe to store the 2 kinds of reliability
    uniqueVids = df.vidName.unique()   
    relDF = pd.DataFrame(columns = ['vidName', 'withinExemplarReliability', 'acrossExemplarReliability'])
    relDF['vidName'] = uniqueVids;
    relDF = relDF.set_index('vidName')
    
    # loop through the videos
    for v in uniqueVids:
        vIdx = np.where(uniqueVids == v)[0][0]
        print('video ' + str(vIdx+1) + ' / ' + str(len(uniqueVids)))
        
        # get the embeddings from all subs for this video
        currVidEmbeddings = df[df['vidName'] == v]
        
        # iteratively calculate split-half reliability for the embeddings
        withinRels = np.zeros((1, nIters)).flatten()
        acrossRels = np.zeros((1, nIters)).flatten()
        for i in range(nIters):
            # within-exemplar reliability
            # ---------------------------
            
            # split the data randomly in half
            ws1 = currVidEmbeddings.sample(frac = 0.5)
            ws2 = currVidEmbeddings.drop(ws1.index)
            
            # average embeddings over the subs in each half
            ws1 = ws1.mean(axis = 0)
            ws2 = ws2.mean(axis = 0)
            
            # correlate the averaged vectors across the halves
            withinRels[i] = ws1.corr(ws2, method = 'spearman')
            
            # across-exemplar reliability
            # ---------------------------
            
            # get all the embeddings from the other exemplar
            currVidSet = v[-1]
            actionName = v.split(currVidSet)[0]
            if currVidSet == '1':
                otherExemplarName = actionName + '2';
            elif currVidSet == '2':
                otherExemplarName = actionName + '1';
            else:
                raise Exception("unexpected video set in video name...")
            
            # get data from the other exemplar:
            otherVidEmbeddings = df[df['vidName'] == otherExemplarName]
            
            # randomly sample from the data for both exemplars:
            as1 = currVidEmbeddings.sample(frac = 0.5)
            as2 = otherVidEmbeddings.sample(frac = 0.5)
            
            # average over the data for each sample:
            as1 = as1.mean(axis = 0)
            as2 = as2.mean(axis = 0)
            
            # correlate the averaged vectors across exemplar samples:
            acrossRels[i] = as1.corr(as2, method = 'spearman')
            
            
        # record the within- and across-exemplar reliability for this vid (averaging over iterations)
        relDF.at[v, "withinExemplarReliability"] = np.mean(withinRels)
        relDF.at[v, "acrossExemplarReliability"] = np.mean(acrossRels)
        
    # return the reliability dataframe
    return relDF;

    # [] make a plot: overlapping histograms
    fig, axes = plt.subplots(1, len(stimSets), figsize=(15, 6))
    
            
            
            
            
  