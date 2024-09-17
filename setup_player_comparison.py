# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 17:24:26 2024

https://lichess.org/api/games/user/CrazyCucumber999?max=100&rated=true&perfType=bullet&clocks=true&evals=true

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

import matplotlib.pyplot as plt

from board_information import get_lucas_analytics, phase_of_game

# conditioned on these variables, how strange are others
DEPENDENT_VAR = ["move_time", "centipawn_loss"]
CONDITIONAL_VAR = ["elo","time_left", "game_phase", "complexity", "win_percentage", "efficient_mobility", "narrowness", "activity"]
cond_type = "ccuccccc"
dep_type = "oc"
def load_density(data_file="data.json"):
    #load data for density estimation
    with open(data_file, "r") as f:
        data = json.load(f)
    
    endog = np.array([to_vec(dic, conditioned=False) for dic in data])
    exog = np.array([to_vec(dic, conditioned=True) for dic in data])
    print(endog.shape)
    print(exog.shape)
    density = sm.nonparametric.KDEMultivariateConditional(endog=endog,exog=exog,dep_type=dep_type, indep_type=cond_type, bw='normal_reference')
    return density

def to_vec(dic, conditioned=False):
    game_phase_dic = {"opening":0, "midgame":1, "endgame":2}
    l = []
    for key in ["elo", "move_time", "time_left", "game_phase", "centipawn_loss", "complexity", "win_percentage", "efficient_mobility", "narrowness", "activity"]:
        if (key in CONDITIONAL_VAR) == conditioned:
            if key == "game_phase":
                l.append(game_phase_dic[dic[key]])
            else:
                l.append(dic[key])
    return l


density = load_density()

STOCKFISH = chess.engine.SimpleEngine.popen_uci('Engines/stockfish_14_x64.exe')

def get_game_statistics(game, elo=None):
    if elo is not None:
        entry = {"elo":elo}
    else:
        entry = {"elo":2000} # template elo
    entry["move_time"] = game.clock() - game.next().next().clock()
    entry["time_left"] = game.clock()
    entry["game_phase"] = phase_of_game(game.board())
    xcompl, xmlr, xemo, xnar, xact = get_lucas_analytics(game.board())
    entry["complexity"] = xcompl
    entry["win_percentage"] = xmlr
    entry["efficient_mobility"] = xemo
    entry["narrowness"] = xnar
    entry["activity"] = xact
    if game.eval() is None:
        analysis = STOCKFISH.analyse(game.board(), chess.engine.Limit(depth=20))
        next_analysis = STOCKFISH.analyse(game.next().board(), chess.engine.Limit(depth=20))
        entry["eval"] = analysis['score'].pov(game.turn()).score(mate_score=2500)
        entry["centipawn_loss"] = max(entry["eval"] - next_analysis["score"].pov(game.turn()).score(mate_score=2500) ,0)
    else:
        entry["eval"] = game.eval().pov(game.turn()).score(mate_score=2500)
        entry["centipawn_loss"] = max(game.eval().pov(game.turn()).score(mate_score=2500) - game.next().eval().pov(game.turn()).score(mate_score=2500) ,0)
    return entry

games_per_player = 10

compare_players = random.sample(os.listdir("PGNs"),5)
compare_dst = [os.path.join("PGNs", x) for x in compare_players]

all_likelihoods = []
names = []
for PGN_file in compare_dst:
    player_file = PGN_file.split("\\")[1]
    print("evaluating the moves from pgn: {}".format(player_file))
    player_name = player_file.split("_")[1]
    
    
    pgn = open(PGN_file, encoding="utf-8")
    likelihoods = []
    games_processed = 0
    games_tried = 0
    elos = []
    while games_processed < games_per_player and games_tried < 100:
        games_tried += 1
        print("Games processed:", games_processed)
        try:
            game = chess.pgn.read_game(pgn)
            if game is None:
                print('game type None, continuing...')
                continue
        except UnicodeDecodeError:
            print('error in parsing game')
            continue
        
        white_player = game.headers["White"]
        black_player = game.headers["Black"]
        
        if white_player == player_name:
            side = chess.WHITE
            elo = float(game.headers["WhiteElo"])
        elif black_player == player_name:
            side = chess.BLACK
            elo = float(game.headers["BlackElo"])
        
        if game.headers["TimeControl"] != "60+0" or game.next().clock() != 60.0 or game.next().next().clock() != 60.0:
            print("Time control is not 60+0, skipping")
            continue
        # Only consider games 10 moves in from each side
        try:
            for i in range(20):
                game = game.next()
        except AttributeError:
            print("Game did not last longer than 10 moves, skipping")
            continue
    
        max_moves = 60
        moves_made = 0
        # now start recording
        if game is None:
            print("Game did not last longer than 10 moves, skipping")
            continue
        elif game.next() is None:
            print("Game did not last longer than 10 moves, skipping")
            continue
        elif game.next().next() is None:
            print("Game did not last longer than 10 moves, skipping")
            continue
        elos.append(elo)
        while game is not None and game.next().next() is not None and moves_made < max_moves:
            if game.turn() != side:
                game = game.next()
                moves_made += 1
                continue
            game_info_dic = get_game_statistics(game, elo=elo)
            exog = to_vec(game_info_dic, conditioned=True)
            endog = to_vec(game_info_dic, conditioned=False)
            likelihood = density.pdf(endog, exog)
            likelihoods.append(likelihood.item())
            game = game.next()
            moves_made += 1
        games_processed += 1
    
    print("Found {} moves from {} games from player {}".format(len(likelihoods), games_processed, player_name))
    if len(elos) > 0:
        average_elo = sum(elos)/len(elos)
    else:
        average_elo = 0
    
    names.append(player_name + "_" + str(average_elo))
    all_likelihoods.append(likelihoods)
    
for i, x in enumerate(all_likelihoods):
    plt.hist(np.log(x),bins=25, alpha=0.7, density=True, label=names[i])

plt.legend(loc="upper left")