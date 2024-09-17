# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 11:21:29 2024

@author: xusem
"""
BASE_LIKELIHOOD_NAME = "base_distributions_1_1" + ".json"

DEPENDENT_VAR = ["move_time", "centipawn_loss", "engine_rank"]
CONDITIONAL_VAR = ["elo","time_left", "complexity", "mat_left", "efficient_mobility", "narrowness", "activity", "eval"]

#get rid of centipawn loss vs eval - too many calculations for not so much information
# get rid of efficient mobility vs centipawn loss
# get rid of activity vs centipawn loss

COMBINATIONS = [{"cond": CONDITIONAL_VAR, "dep": DEPENDENT_VAR, "dep_type": "oco", "cond_type":"cccccccc"}, 
                {"dep":["centipawn_loss", "engine_rank"], "cond":CONDITIONAL_VAR, "dep_type": "co", "cond_type":"cccccccc"},
                {"dep":["centipawn_loss", "move_time"], "cond":CONDITIONAL_VAR, "dep_type": "co", "cond_type":"cccccccc"},
                {"dep":["engine_rank", "move_time"], "cond":CONDITIONAL_VAR, "dep_type": "oo", "cond_type":"cccccccc"},
                {"dep":["centipawn_loss"], "cond":["elo","complexity"], "dep_type": "c", "cond_type":"cc"},
                {"dep":["centipawn_loss"], "cond":["elo","time_left"], "dep_type": "c", "cond_type":"cc"},
                {"dep":["engine_rank"], "cond":["elo","time_left"], "dep_type": "c", "cond_type":"cc"},
                {"dep":["engine_rank"], "cond":["elo", "eval"], "dep_type": "o", "cond_type":"cc"},
                {"dep":["engine_rank"], "cond":["elo", "complexity"], "dep_type": "o", "cond_type":"cc"},
                {"dep":["engine_rank"], "cond":["elo", "efficient_mobility"], "dep_type": "o", "cond_type":"cc"},
                {"dep":["engine_rank"], "cond":["elo", "activity"], "dep_type": "o", "cond_type":"cc"},
                {"dep":["move_time"], "cond":["elo", "eval"], "dep_type": "o", "cond_type":"cc"},
                {"dep":["move_time"], "cond":["elo", "complexity"], "dep_type": "o", "cond_type":"cc"},
                {"dep":["move_time"], "cond":["elo", "efficient_mobility"], "dep_type": "o", "cond_type":"cc"},
                {"dep":["move_time"], "cond":["elo", "activity"], "dep_type": "o", "cond_type":"cc"},
                ]

CATEGORY_RANGES = {
    "move_time": [0,1,2,3,4,5],
    "centipawn_loss": [x*5 for x in range(50)],
    "engine_rank": [1,2,3,4,5,6],
    "elo": [x*25 + 2200 for x in range(40)],
    "complexity": [5*x for x in range(20)],
    "mat_left": [20 + 5*x for x in range(14)],
    "efficient_mobility": [2.5*x for x in range(20)],
    "narrowness": [2.5*x for x in range(20)],
    "activity": [1.25*x for x in range(20)],
    "eval": [(x-20)*40 for x in range(41)],
    "time_left": [x+1 for x in range(15)],
    }