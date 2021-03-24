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


# %% Regression from human judgments

# %% Compare to ImageNet representations

# [] load in imageNet features (run on each video frame and then averaged over the frames)

