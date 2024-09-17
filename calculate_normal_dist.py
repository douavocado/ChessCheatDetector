# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 23:31:31 2024

@author: xusem
"""

import statsmodels.api as sm
import json
import numpy as np
import os
import chess
import chess.pgn
import random
from tqdm import tqdm


from constants import DEPENDENT_VAR, CONDITIONAL_VAR, COMBINATIONS, BASE_LIKELIHOOD_NAME

# conditioned on these variables, how strange are others

def load_density(data_file="data.json", dep=DEPENDENT_VAR, cond=CONDITIONAL_VAR, dep_type="cco", cond_type="ccccccc"):
    #load data for density estimation
    with open(data_file, "r") as f:
        data = json.load(f)
    
    endog = np.array([to_vec(dic, dep) for dic in data])
    exog = np.array([to_vec(dic, cond) for dic in data])
    density = sm.nonparametric.KDEMultivariateConditional(endog=endog,exog=exog,dep_type=dep_type, indep_type=cond_type, bw='normal_reference')
    return density

def to_vec(dic, keys_wanted):
    l = []
    for key in keys_wanted:
        l.append(dic[key])
    return l


data_file = os.path.join("Databases", "all_data.json")
STOCKFISH = chess.engine.SimpleEngine.popen_uci('Engines/stockfish_14_x64.exe')

with open(data_file, "r") as f:
    all_data = json.load(f)

#load densities
all_densities = []
for combination in COMBINATIONS:
    density = load_density(data_file, dep=combination["dep"], cond=combination["cond"], dep_type=combination["dep_type"], cond_type=combination["cond_type"])
    all_densities.append(density)

print("densities loaded")

all_likelihoods = []

if len(all_data) > 10000:
    subset_data = random.sample(all_data,10000) # for lesser computation
else:
    subset_data = all_data

for entry in tqdm(subset_data):
    entry_likelihoods = []
    for j, combination in enumerate(COMBINATIONS):
        endog = np.array(to_vec(entry, combination["dep"]))
        exog = np.array(to_vec(entry, combination["cond"]))
        # compute likelihood
        
        likelihood = all_densities[j].pdf(endog, exog)
        entry_likelihoods.append(likelihood.item())
    all_likelihoods.append(entry_likelihoods)

file_name = os.path.join("Databases", BASE_LIKELIHOOD_NAME)
with open(file_name, 'w') as fout:
    json.dump(all_likelihoods, fout)
