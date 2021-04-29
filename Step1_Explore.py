# -*- coding: utf-8 -*-
"""
Leyla Tarhan
lytarhan@gmail.com
3/2021

This script contains code to interactively explore:
    (1) the verbal descriptions that were input into the BERT model 
    (a deep net trained for NLP) and 
    
    (2) the resulting feature embeddings extracted from the BERT model
    
"""
# %% libraries and helpers

# custom helper functions
from utils import *

import pandas as pd
from pathlib import Path

# %% load in the data

# descriptions of each video
descriptionsPath = Path("Data") / "ActionDescriptions.csv"
descriptions = pd.read_csv(descriptionsPath)

# full BERT feature embeddings
embeddingsPath = Path('Data') / 'DescriptionEmbeddings.csv'
embeddings = pd.read_csv(embeddingsPath)

print('Loaded data: verbal descriptions and their feature embeddings.')

# %% Explore the descriptions of a set of action videos

# run this cell to call a helper function -- it will print out the descriptions collected for any video. 
printDescriptions(descriptions)
    
# %% how reliable are the BERT embeddings across subjects?

# run this cell to plot how similar the embeddings are across subjects. 
# similarity is computed as the euclidean distance between any 2 subjects' embeddings, averaged across videos. 
printInterSubReliability(embeddings)

# there are a couple of apparent outliers in this data (columns that stand out in the heatmap).
# exclude them before moving on.
excludeFlag = 1;
subsToExclude = ['Sub11', 'Sub7', 'Sub27'];
if excludeFlag:
    embeddings = excludeOutliers(embeddings, subsToExclude)

# %% how video-specific are the BERT embeddings?

# the stimuli are split into 2 sets of videos. Each set contains 1 video for each of 60 everyday actions. 
# So, across the video sets there are 2 exemplars of the same action. The exemplars vary in a lot of low-level visual properties
# -- such as the actor's appearance, the direction they're moving, the background, etc. 
# They might unintentionally vary in some higher-level features, too -- e.g., maybe a professional runner has different goals than an amateur.

# are the NLP features more similar across descriptions of the same video exemplar than across descriptions of different exemplars of the same action?
# answering this doesn't necessarily tell us whether these features are capturing something visual or high-level (see above).
# but, it's worth knowing the answer in order to understand whether it's reasonable to collapse across exemplars.
# It could also inform follow-up experiments to try to hone in more on whether the features are visual or higher-level. 

# approach: compare split-half reliability within the same video vs. across exemplars of the same action

reliabilityResults = withinBetweenReliability(embeddings);


# %% which videos have the most / least similar BERT embeddings?

# [x] average embeddings over subs
# [x] get embedding similarity among all videos
# [x] print out a sorted bar plot of pair similarity
# [] show key frames for the n most and n least similar vids -- maybe the user specifies how many? key frames should be labeled I think

embeddingSimilarities(embeddings, 5)