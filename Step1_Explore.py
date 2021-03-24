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
# [] add these in

import pandas as pd
from pathlib import Path

# %% load in the data

# descriptions of each video
descriptionsPath = Path("Data") / "ActionDescriptions.csv"
descriptions = pd.read_csv(descriptionsPath)

# [] full BERT feature embeddings

# %% Explore the descriptions of a set of action videos

# [] write a helper function: 
    # prints out a list of the vid names
    # asks you to enter a vid name (or quit)
    # prints out all the descriptions of that vid
    # asks you to do it again (or quit)
    
# %% how reliable are the BERT embeddings across subjects?


# %% how video-specific are the BERT embeddings?
# compare split-half reliability within the same video vs. across exemplars of the same action


# %% which videos have the most / least similar BERT embeddings?

