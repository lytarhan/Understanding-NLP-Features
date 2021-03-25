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

# [] full BERT feature embeddings

# %% Explore the descriptions of a set of action videos

# run this cell to call a helper function -- it will print out the descriptions collected for any video. 
printDescriptions(descriptions)
    
# %% how reliable are the BERT embeddings across subjects?


# %% how video-specific are the BERT embeddings?
# compare split-half reliability within the same video vs. across exemplars of the same action


# %% which videos have the most / least similar BERT embeddings?

