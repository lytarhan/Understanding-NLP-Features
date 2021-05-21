# -*- coding: utf-8 -*-
"""
Leyla Tarhan
lytarhan@gmail.com
3/2021

This script contains code to try to understand what kinds of information is 
being captures by NLP feature embeddings from descriptions of action videos.

This is done by comparing feature embeddings extracted from a BERT model to 
better-understood features, in 2 ways:
    (1) regression - use 3 better-understood kinds of information about the 
    actions (judgments about the videos' visual appearance, the actors' 
    movements, and the actors' goals) to explain the BERT feature embeddings, 
    and see how well each of them does.
    
    (2) compare the structure captured by the BERT feature embeddings to the 
    structure captured by ImageNet, which is more directly tied to image 
    properties rather than semantics.
    
"""

# %% libraries and helpers

from pathlib import Path
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from utils import *

pd.set_option('display.max_columns', 10)  

# %% load in the data

# (1) full BERT feature embeddings for each video
embeddingsPath = Path('Data') / 'DescriptionEmbeddings.csv'
embeddings = pd.read_csv(embeddingsPath)

# remove outliers and then average over subs to get a single embedding vector per video
subsToExclude = ['Sub11', 'Sub7', 'Sub27'];
embeddings = excludeOutliers(embeddings, subsToExclude)
embeddingsOverSubs = embeddings.groupby('vidName').mean();

# no human data for video set 2, so pull out the embeddings for just set 1
embeddingsOverSubs['SetNum'] = [vidname[-1] for vidname in embeddingsOverSubs.index]
set1Embeddings = embeddingsOverSubs[embeddingsOverSubs['SetNum'] == '1'];

print('Loaded data: feature embeddings for descriptions of each action video.')


# (2) human similarity judgments for these same videos, along 3 different dimensions
judgmentsPath = Path('Data') / 'HumanSimilarityJudgments.csv'
judgments = pd.read_csv(judgmentsPath)
judgments['vid1'] = [vn.replace('_', '-') for vn in judgments['vid1']]
judgments['vid2'] = [vn.replace('_', '-') for vn in judgments['vid2']]

print('Loaded data: human similarity judgments for all action videos in set 1.')


# transform embeddings into similarities (euclidean distance between each pair of videos' embedding vectors)
embeddingRDM = getEmbeddingRDM(set1Embeddings, judgments)
print('formatted feature embeddings into RDM.')


# %% Regression from human judgments




# %% Compare to ImageNet representations

# [] load in imageNet features (run on each video frame and then averaged over the frames)

