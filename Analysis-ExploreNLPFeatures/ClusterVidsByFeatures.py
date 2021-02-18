# -*- coding: utf-8 -*-
"""
# Leyla Tarhan 
# 7/2020

# run a few clustering analyses on the natural language semantics features 
# collected through the BERT pipeline. In all of these, we're clustering the 
# *videos* based on the features (because videos are more interpretable)

# all of these analyses will be done over the PC'd feature space

# first, compare the features across video sets to get an idea of how generalizable they are, and also how abstract

# then, do clustering analyses to look for structure in the features:
    # ...hierarchical clustering (with a dendrogram visualization)
    # ...MDS (then look at the 2- or 3-d plot, where each video's point is labeled)
    # ...k-means clustering (then visualize the profiles for each cluster)
    # ...tSNE (this didn't reveal anything though, so didn't pursue it very far)


"""

# %% Clean up

try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass

# %% libraries

# loading files:
from scipy.io import loadmat    
import os 

# formatting data:
import pandas as pd
import numpy as np
from scipy import stats
from sklearn import preprocessing
from sklearn.metrics import pairwise_distances

# clustering:
import scipy.cluster.hierarchy as sch
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.manifold import TSNE

# plotting:
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.cm as cm
import seaborn as sns


# %% setup

# directory with the BERT features:
bertDir = "C:\\Users\Leyla\Dropbox (KonkLab)\Research-Tarhan\Project - ActionMapping - Videos\Experiments - Behavioral\Experiment - SemanticEmbeddings-Full\Data\Feature Models"

sets = [1, 2]
nVids = 60 # vids per set


# %% read in the features for each video set
    

# set up a dict that will store the data from both sets. I think this is better
#  than a dataframe, bc there are slightly different number of PC'd features 
# for the 2 sets and those PC's also don't necessarily correspond across sets.
dataDict = {};

# loop through the sets and load in the data:
for s in sets:
    # read in the .mat file with the PC'd features
    mn = os.path.join(bertDir, 'Set' + str(s) + '-BERTFeatureModel.mat')
    vidNames = loadmat(mn)['itemLabels']; # numpy array
    pcFeatures = loadmat(mn)['featureMatrix']; # numpy array
    # pcFeatures.shape() # vids x features
    assert pcFeatures.shape[0] == nVids, 'check dimensions on pcFeatures (should be vids x PCs)';
    
    # convert to dataframe (to combine the vid names & features in 1 object):
    nFeatures = pcFeatures.shape[1]
    colNames = ['PC' + str(n) for n in range(1, nFeatures+1)] # column names
    df = pd.DataFrame(data = pcFeatures, index = vidNames, columns = colNames);        
    
    # convert dataframe to dict (to store data from both sets in a dict):
    dataDict['Set' + str(s)] = df.to_dict();
    # to convert back: df2 = pd.DataFrame.from_dict(dataDict['Set1'])
    
print('\n\nloaded all data to a dict.')

# %% compare feature structure across video sets
# correlate the RDMS (vids x vids) for video sets 1 and 2
# gives an idea of how abstract the features are

vidNamesDict = {} # setup to store the video names for each set (to make sure the order is the same)
rdmsDict = {}

# make an RDM out of each set (correlation distance)
for s in list(dataDict.keys()):
    # pull out the data -> data frame
    df = pd.DataFrame.from_dict(dataDict[s])
    
    # sort it alphabetically by video name
    dfSort = df.sort_index(axis = 0) 
    
    # pull out and clean the video names:
    setNum = s.strip('Set')
    vidNamesDict[s] = [name.replace(setNum, '').lower().strip() for name in dfSort.index]
    if 'sleeping-sleep' in vidNamesDict[s]:
        sleepIdx = vidNamesDict[s].index('sleeping-sleep')
        vidNamesDict[s][sleepIdx] = 'sleep-sleep'
    
    # calculate the RDM
    rdmsDict[s] = {}
    rdm = pairwise_distances(dfSort.values, metric = 'correlation') # full, square matrix
    assert rdm.shape == (nVids, nVids), "RDM is unexpected shape (should be vids x vids)"
    rdmsDict[s]['matrix'] = rdm
    
    # get just the upper triangle minus the diagonal and store it
    rdmTriIdx = np.triu(np.ones(rdm.shape), k = 1).astype(np.bool) # booleans: true if in the upper triangle minus diagonal, false if not
    rdmTri = rdm[rdmTriIdx] # pull out just the triangle as a vector
    rdmsDict[s]['triangle'] = rdmTri
    
    
# check that order matches across sets
assert all(vidNamesDict['Set1'][i] == vidNamesDict['Set2'][i] for i in range(nVids))

# correlate the RDMs' lower triangles
r = stats.spearmanr(rdmsDict['Set1']['triangle'], rdmsDict['Set2']['triangle'])[0]

# print them out
fig, axes = plt.subplots(1, 2, figsize = (10, 5))
fig.suptitle('Cross-sets RDM correlation (rho) = ' + str(round(r, ndigits = 2)))
ax1 = sns.heatmap(rdmsDict['Set1']['matrix'], 
                  square = True, 
                  ax=axes[0], 
                  xticklabels=False, 
                  yticklabels=False, 
                  cbar_kws={"shrink": .6, "label": 'correlation distance'})
ax1.set_title('Set1: \n corr distance matrix')

ax2 = sns.heatmap(rdmsDict['Set2']['matrix'], 
                  square = True, 
                  ax=axes[1], 
                  xticklabels=False, 
                  yticklabels=False,
                  cbar_kws = {'shrink': .6, 'label': 'correlation distance'})
ax2.set_title('Set2: \n corr distance matrix')



# %% hierarchical clustering

# loop through the video sets
for s in list(dataDict.keys()):
    # pull out the data for this set and convert back to a dataframe
    df = pd.DataFrame.from_dict(dataDict[s])
    
    # plot the dendrogram based on linkage using the Ward method (minimizes the variance within each cluster): 
    figure(num=None, figsize=(13, 14), dpi=80, facecolor='w', edgecolor='k')
    dendrogram = sch.dendrogram(sch.linkage(df.values, method  = "ward"), labels=df.index, leaf_font_size = 10, orientation="left"); # plot & linkage

    # modify the plot:
    plt.title(s)
    plt.xlabel('Videos')
    plt.ylabel('Euclidean distance')
    plt.show()
    
    # make sure the clustering worked in the right way (should have 1 leaf node per video):
    assert len(dendrogram['ivl']) == nVids, "check that you're clustering vids based on features, not vice versa."

print('\n\nhierarchical clustering: DONE.')

# %% Multi-dimensional scaling
# try to visualize the structure of the videos when they're projected into a lower-dimensional space based on the BERT features
# (very similar to the hclust method, but could give a better sense of the data's geometric "shape")

# set up max # of dimensions to consider:
maxDims = 10; # obviously not going to visualize 10d, but easier to see an elbow this way

# loop through the video sets
for s in list(dataDict.keys()):
    # pull out the data for this set and re-format as a dataframe:
    df = pd.DataFrame.from_dict(dataDict[s])
    
    # MDs, projecting into a range of dimensions:
    stress = [];
    for d in range(1, maxDims+1):
        mds = MDS(n_components=d, metric=True, dissimilarity='euclidean'); # set up the MDS object
        X_transformed = mds.fit_transform(df.values); # vids x mds dimensions
        assert X_transformed.shape == (nVids, d), "check that you're clustering vids with MDS, not features"
        
        # record the stress:
        stress.append(mds.stress_) # sum of squared distance of the disparities and the distances for all constrained points)
    
    # plot stress x projected dimensions:
    figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(range(1, maxDims+1), stress)
    plt.title(s)
    plt.xlabel('MDS dimensions')
    plt.ylabel('Stress')
    plt.show()
    
    
    # pause to assess the stress plot, get input on # of dimensions:
    finalDims = int(input(s + ': how many dimensions for MDS? '))
    
    # MDS with # of dims indicated above:
    mds = MDS(n_components = finalDims, metric = True, dissimilarity = 'euclidean');
    X_transformed = mds.fit_transform(df.values);
    assert X_transformed.shape == (nVids, finalDims), "check the final MDS dimensions"
    
    # plot vids in MDS space
    if finalDims == 3: # 3d plot
        fig = plt.figure(figsize = (10, 7)) 
        ax = plt.axes(projection ="3d") 
        x = X_transformed[:, 0]
        y = X_transformed[:, 1]
        z = X_transformed[:, 2]
        
        # plot each point with a label:
        for a in range(len(x)):
            ax.scatter(x[a], y[a], z[a], color = "green")
            ax.text(x[a], y[a], z[a], df.index[a], size=8, zorder=1, color='k')
        
        plt.title(s + ': 3d MDS projection')
        
        # [] rotate the plot for better viewing (doesn't keep updating -- maybe better with a notebook?):
        # for angle in range(0, 360):
        #     ax.view_init(30, angle)
        #     plt.draw()
        #     plt.pause(.001)
        
    else: # 2d plot
        fig = plt.figure(figsize = (10, 7))
        ax = fig.add_subplot(111)
        x = X_transformed[:, 0];
        y = X_transformed[:, 1];
        
        # plot each point with a label:
        for a in range(len(x)):
            ax.scatter(x[a], y[a], color = "green")
            ax.text(x[a], y[a], df.index[a], size = 8, color = 'k')
        
        plt.title(s + ': 2d MDS projection')

print('\n\nMDS: DONE.')
    
# %% k-means clustering
# cluster the features based on the profile of their values over the videos. Then, examine each cluster's profile to see if they're interpretable
# (interpretability will depend upon the videos we feed in, of course)

# max k-value (# of clusters) to consider:
maxK = 20;
    
# loop through the video sets
for s in list(dataDict.keys()):
    # pull out the data and re-format as an array
    df = pd.DataFrame.from_dict(dataDict[s]) # dict -> data frame
    featureArray = df.values.transpose() # data frame -> transposed np array (rows = features, cols = vids)
    assert featureArray.shape[1] == nVids, "check featureArray's dimensions -- should be features x vids"
    
    
    # try to figure out a good k -- consider k = 2 to some max number
    avgSilh = []
    semSilh = []
    for k in range(2, maxK+1):
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
        labels = kmeans.fit_predict(featureArray) # compute cluster centers
        
        # get the silhouette values:
        silhouette_avg = silhouette_score(featureArray, labels) # avg. over all values
        silh_samples = silhouette_samples(featureArray, labels) # silhouette coeff for each sample

        avgSilh.append(silhouette_avg) # store it for a summary plot
        semSilh.append(stats.sem(silh_samples))
        
        
    # summary plot: avg silhouette coefficient for each k
    # error bars = standard error
    fig = plt.figure(figsize = (10, 7))
    ax = fig.add_subplot(111)
    plt.errorbar(range(2, maxK+1), avgSilh, semSilh)
    plt.title(s + ': silhouette summary')
    plt.xlabel('number of clusters')
    plt.ylabel('average silhouette coefficient')
    ax.set_xticks(range(2, maxK+1))
    plt.show()
    
    
    # user input: what k to use?
    finalK = int(input(s + ': how many clusters? '))
    
    # get cluster centers & assignments for this solution
    kmeans = KMeans(n_clusters=finalK, init='k-means++', max_iter=300, n_init=10, random_state=0)
    labels = kmeans.fit_predict(featureArray) # compute cluster centers
    centers = kmeans.cluster_centers_; # k x vids
    
    # plot each cluster's profile
    for c in range(finalK):
        # get this cluster's center/profile
        currProfile = centers[c,] # 1 value for each vid
        
        # how many features in this cluster?
        nMembers = len(labels[labels == c])
        
        # plot the profile in a unique color
        fig = plt.figure(figsize = (7, 15))
        ax = fig.add_subplot(111)
        currColor = cm.nipy_spectral(float(c) / finalK)
        plt.barh(df.index, currProfile, color = currColor)
        plt.title(s + ': cluster profile ' + str(c+1) + '/' + str(finalK) + ' (' + str(nMembers) + ' features)')
        plt.xlabel('centroid feature value')
        ax.yaxis.set_ticks_position('right')
        plt.show()
    

print('\n\nK-Means: DONE.')

# %% t-SNE
# Sort of like MDS: project the videos into a 2d map and see which videos are in clusters together.

# loop through the sets
for s in list(dataDict.keys()):
    # pull out the data for this set, and turn into an array
    df = pd.DataFrame.from_dict(dataDict[s]) # dict -> data frame
    scaledVidArray = preprocessing.scale(df.values) # data frame -> normalized np array (vids x features)
    # now each feature has a mean of 0 and standard deviation of 1
    
    
    # tSNE
    perplex = 30
    tsneEmbed = TSNE(n_components=2, verbose=1, perplexity=perplex, n_iter=1000).fit_transform(scaledVidArray)
    # result: embedding in a 2-d map
    # how do you decide on perplexity?
    #...default = 30
    #...should be < # observations (60 in this case)
    #...choose perplexity based on plots? (how do you know what a "good" plot looks like??)
    #...best stability or lowest KL divergence?
    #...unclear how to get these values, so instead I'll just go with the defaults and check out the plot
      
    
    # plot it
    fig = plt.figure(figsize = (10, 10))
    ax = fig.add_subplot(111)
    ax.scatter(tsneEmbed[:,0], tsneEmbed[:,1])
    plt.title(s + ': tSNE embedding with perplexity = ' + str(perplex))
    plt.xlabel('dimension 1')
    plt.ylabel('dimension 2')
    plt.show()
    
    # if there were evident clusters, could then apply a clustering algorithm to this 2d embedding and see which vids were clustered together -- but in this case, it's all just uniformally spread


print('\n\ntSNE: DONE.')

