# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 23:48:06 2024

@author: xusem
"""

import statsmodels.api as sm
import json
import numpy as np
import os
import chess
import chess.pgn
import random
import math
from scipy.stats import ks_2samp
import seaborn as sns

import matplotlib.pyplot as plt

from constants import DEPENDENT_VAR, CONDITIONAL_VAR, COMBINATIONS, BASE_LIKELIHOOD_NAME, CATEGORY_RANGES

STOCKFISH = chess.engine.SimpleEngine.popen_uci('Engines/stockfish_14_x64.exe')

TARGET_PGN = os.path.join("new_PGNs", random.choice(os.listdir("new_PGNs")))
#TARGET_PGN = "testPGNs\\lichess_CrazyCucumber999_2024-09-07.pgn"
#TARGET_PGN = "testPGNs\\lichess_DrActuallyMe_2024-09-09.pgn"

games_used = 25

# conditioned on these variables, how strange are others

def load_density(data_file="data.json", dep=DEPENDENT_VAR, cond=CONDITIONAL_VAR, dep_type="cco", cond_type="ccccccc"):
    #load data for density estimation
    with open(data_file, "r") as f:
        data = json.load(f)
    
    endog = np.array([to_vec(dic, dep) for dic in data])
    exog = np.array([to_vec(dic, cond) for dic in data])
    density = sm.nonparametric.KDEMultivariateConditional(endog=endog,exog=exog,dep_type=dep_type, indep_type=cond_type, bw='normal_reference')
    return density

def load_density_from_list(list_dic, dep=DEPENDENT_VAR, cond=CONDITIONAL_VAR, dep_type="cco", cond_type="ccccccc"):
    #load data for density estimation from list of dictionaries (entries)
    
    endog = np.array([to_vec(dic, dep) for dic in list_dic])
    exog = np.array([to_vec(dic, cond) for dic in list_dic])
    density = sm.nonparametric.KDEMultivariateConditional(endog=endog,exog=exog,dep_type=dep_type, indep_type=cond_type, bw='normal_reference')
    return density

def load_compare_density(data_file="base_distdata_3000.json"):
    with open(data_file, "r") as f:
        likelihoods = json.load(f)
    return likelihoods

def to_vec(dic, keys_wanted):
    l = []
    for key in keys_wanted:
        l.append(dic[key])
    return l

def get_lucas_analytics(board):
    ''' Given a chess position, give the following stats about the position:
        Complexity: how complex the position is. from 0 to 100
        Win probability (of turn's side). from 0 to 100
        Efficient mobility: how easy it is to play without making mistakes. from 0 to 50
        Narrowness: how open the position is. from 0 to 50
        Piece activity (of turn's side). from 0 to 25'''
    
    mat_dic = {1:1, 2:3.1, 3:3.5, 4:5.5, 5:9.9, 6:3}
    #calculate constants
    xpie = len(board.piece_map()) # number of pieces alive on the board
    xpiec = sum([len(board.pieces(x, board.turn)) for x in range(1,7)]) # number of pieces alive of board turn colour
    xmov = board.legal_moves.count() # number of legal moves for turn player
    xpow = sum([len(board.pieces(x, board.turn))*mat_dic[x] for x in range(1,7)]) # sum of material on turn side
    xmat = xpow + sum([len(board.pieces(x, not board.turn))*mat_dic[x] for x in range(1,7)]) # sum of all material on board
    
    analysis = STOCKFISH.analyse(board, chess.engine.Limit(time=0.02), multipv=18)
    moves = [entry["pv"][0].uci() for entry in analysis]
    evals = [entry['score'].pov(board.turn).score(mate_score=2500) for entry in analysis]
    move_evals = list(zip(moves, evals))
    top_moves = sorted(move_evals, key= lambda x: x[1], reverse=True)
    xeval = max(evals) # evaluation of position (in centipawns)
    xgmo = len([x for x in evals if x + 100 >= xeval]) # number of good moves in position
    
    # complexity
    xcompl = xgmo * xmov * xpie * xmat / (400 * xpow)
    # win prob
    xmlr = (math.tanh(xeval/(2*xmat)) + 1) * 50
    # efficient mobility
    xemo = 100*(xgmo - 1)/xmov
    # narrowness
    xnar = 10 * (xpie**2) * xmat / (4 * (xgmo**0.5)* (xmov**0.5)* xpiec * xpow)
    # piece activity
    xact = 40 * (xgmo**0.5)* (xmov**0.5) * xpow / (xpie * xmat)
    
    
    #clipping
    xcompl = min(xcompl, 100)
    xemo = min(xemo, 50)
    xnar = min(xnar, 50)
    xact = min(xact, 25)
    return xcompl, xmlr, xemo, xnar, xact, top_moves, xmat

""" 
Variables we want to record:
player elo (the player making the move)
time spent on move
time left
eval of game from turn perspective
centipawn loss of move
engine rank if in top 5, else 6
lucas analytics including:
    complexity
    efficient mobility
    narrowness of position
    piece activity
    material left on board

"""

def get_game_statistics(game, elo=None):
    if elo is not None:
        entry = {"elo":elo}
    else:
        entry = {"elo":2000} # template elo
    entry["move_time"] = game.clock() - game.next().next().clock()
    entry["time_left"] = game.clock()
    xcompl, xmlr, xemo, xnar, xact, top_moves, xmat = get_lucas_analytics(game.board())
    entry["complexity"] = xcompl
    entry["efficient_mobility"] = xemo
    entry["narrowness"] = xnar
    entry["activity"] = xact
    entry["mat_left"] = xmat
    top_ucis = [x[0] for x in top_moves]
    move_played = game.next().move.uci()
    if move_played in top_ucis[:5]:
        entry["engine_rank"] = top_ucis.index(move_played) + 1
        
    else:
        entry["engine_rank"] = 6
    if game.eval() is None:
        analysis = STOCKFISH.analyse(game.board(), chess.engine.Limit(time=0.02))
        next_analysis = STOCKFISH.analyse(game.next().board(), chess.engine.Limit(time=0.02))
        entry["eval"] = analysis['score'].pov(game.turn()).score(mate_score=2500)
        entry["centipawn_loss"] = max(entry["eval"] - next_analysis["score"].pov(game.turn()).score(mate_score=2500) ,0)
    else:
        entry["eval"] = game.eval().pov(game.turn()).score(mate_score=2500)
        entry["centipawn_loss"] = max(game.eval().pov(game.turn()).score(mate_score=2500) - game.next().eval().pov(game.turn()).score(mate_score=2500) ,0)
    return entry

def KL(P,Q):
    """ Epsilon is used here to avoid conditional code for
checking that neither P nor Q is equal to 0. """
    epsilon = 0.00001
    
    # You may want to instead make copies to avoid changing the np arrays.
    P = P+epsilon
    Q = Q+epsilon
    
    divergence = np.sum(P*np.log(P/Q))
    return divergence

data_file = os.path.join("Databases", "all_data.json")

# load all densities
all_densities = [load_density(data_file, dep=comb["dep"], cond=comb["cond"], dep_type=comb["dep_type"], cond_type=comb["cond_type"]) for comb in COMBINATIONS]

player_file = TARGET_PGN.split("\\")[1]
print("evaluating the moves from pgn: {}".format(player_file))
player_name = player_file.split("_")[1]
pgn_date = player_file.split("_")[2]

pgn = open(TARGET_PGN, encoding="utf-8")
all_likelihoods = []
games_processed = 0
games_tried = 0
elos = []
test_list_dics = []
while games_processed < games_used and games_tried < 100:
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
    # load correct base_dist and density
    densities = []
    if elo < 2200:
        print("player elo not high enough: {}. Skipping game".format(elo))
        continue
    
    while game is not None and game.next().next() is not None and moves_made < max_moves:
        if game.turn() != side:
            game = game.next()
            moves_made += 1
            continue
        game_info_dic = get_game_statistics(game, elo=elo)
        test_list_dics.append(game_info_dic)
        entry_likelihoods = []
        
        for i, density in enumerate(all_densities):
            comb = COMBINATIONS[i]
            exog = to_vec(game_info_dic, comb["cond"])
            endog = to_vec(game_info_dic, comb["dep"])
            entry_likelihoods.append(density.pdf(endog, exog).item())
        
        all_likelihoods.append(entry_likelihoods)
        game = game.next()
        moves_made += 1
    games_processed += 1

average_elo = sum(elos)/len(elos)

# Create directory for player report
dir_name = player_name + "_" + str(round(average_elo)) + "_" + pgn_date
player_path = os.path.join("Player_Reports", dir_name)
if not os.path.isdir(player_path):
    os.mkdir(player_path)

base_likelihoods = load_compare_density(os.path.join("Databases", BASE_LIKELIHOOD_NAME))
# now analyse
all_likelihoods = np.array(all_likelihoods)
base_likelihoods = np.array(base_likelihoods)

fig, axs = plt.subplots(4, 4)
fig.set_figheight(25)
fig.set_figwidth(35)
for i, combination in enumerate(COMBINATIONS):
    ax_i = i // 4
    ax_j = i % 4
    # compute base distribution histogram
    base_values = base_likelihoods[:,i]
    test_values = all_likelihoods[:,i]
    base_heights, chosen_bins, _ = axs[ax_i, ax_j].hist(base_values, bins="sturges", alpha=0.7, density=True, label="base")
    test_heights, _, _ = axs[ax_i, ax_j].hist(test_values, bins=chosen_bins, alpha=0.7, density=True, label=player_name)
    axs[ax_i, ax_j].set_xlabel("cond:" + str(combination["cond"]))
    axs[ax_i, ax_j].set_ylabel("dep:" + str(combination["dep"]))
    kl_div = round(KL(test_heights, base_heights),5)
    ks_test = round(ks_2samp(test_values, base_values).pvalue,5)
    axs[ax_i, ax_j].set_title("KL_div: {}, K-S p value: {}".format(kl_div, ks_test))
    axs[ax_i, ax_j].legend(loc="upper right")

fig.suptitle("Data evaluated from {} moves from {} games".format(all_likelihoods.shape[0], games_processed))
fig.savefig(os.path.join(player_path, player_name + "_" + str(round(average_elo)) +"_eval.png"))

# Now create density plots for all 2 dimensional exog + 1 dimensional 
# We don't plot but use the average elo of player for each of these plots
# So for example cond = ["elo", "eval"] dep = ["engine_rank"] we plot 
# eval on x axis, engine rank on y axis and colour to display heatmap
# and similarly for base density
print("Creating Density Plots...")

for i, combination in enumerate(COMBINATIONS):
    if len(combination["cond"]) == 2 and len(combination["dep"]) == 1:
        pass
    else:
        continue
    # first create player density
    test_density = load_density_from_list(test_list_dics, dep = combination["dep"], cond=combination["cond"], dep_type=combination["dep_type"], cond_type=combination["cond_type"])
    # identify base density
    base_density = all_densities[i]
    # set limits for heatmap
    real_cond = [x for x in combination["cond"] if x != "elo"]
    assert len(real_cond) == 1
    x_name = real_cond[0]
    y_name = combination["dep"][0]
    x_range = CATEGORY_RANGES[x_name]
    y_range = CATEGORY_RANGES[y_name]
    
    test_heatmap_vals = np.zeros((len(x_range), len(y_range)))
    base_heatmap_vals = np.zeros((len(x_range), len(y_range)))
    for i_x,x in enumerate(x_range):
        for i_y, y in enumerate(y_range):
            if combination["cond"][0] == "elo":
                test_exog = [average_elo, x]
            elif combination["cond"][1] == "elo":
                test_exog = [x, average_elo]
            else:
                raise Exception("Elo not found in combination {}. This is required for heatmap plots".format(combination["cond"]))
            test_heatmap_vals[i_x,i_y] = test_density.pdf([y], test_exog).item()
            base_heatmap_vals[i_x,i_y] = base_density.pdf([y], test_exog).item()
    
    # save figure
    fig, axs = plt.subplots(1, 2)
    fig.set_figheight(10)
    fig.set_figwidth(25)
    
    g1 = sns.heatmap(test_heatmap_vals.transpose(), cmap="YlGnBu", yticklabels=y_range, xticklabels=x_range, ax=axs[0])
    g1.set_xlabel("cond:" + x_name)
    g1.set_ylabel("dep:" + y_name)
    g1.set_title("Distribution for {}".format(player_name))
    g2 = sns.heatmap(base_heatmap_vals.transpose(), cmap="YlGnBu", yticklabels=y_range, xticklabels=x_range, ax=axs[1])
    g2.set_xlabel("cond:" + x_name)
    g2.set_ylabel("dep:" + y_name)
    g2.set_title("Base distribution")
    
    fig.suptitle("Data evaluated from {} moves from {} games".format(all_likelihoods.shape[0], games_processed))
    fig.savefig(os.path.join(player_path, player_name + "_" + str(round(average_elo)) + "_" + x_name + "_vs_" + y_name +"_eval.png"))
    
    