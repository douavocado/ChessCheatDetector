# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 12:02:29 2024

@author: xusem
"""
import os
import chess
import chess.pgn
import math
import json
from tqdm import tqdm


STOCKFISH = chess.engine.SimpleEngine.popen_uci('Engines\\stockfish_14_x64.exe')

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
        analysis = STOCKFISH.analyse(game.board(), chess.engine.Limit(depth=20))
        next_analysis = STOCKFISH.analyse(game.next().board(), chess.engine.Limit(depth=20))
        entry["eval"] = analysis['score'].pov(game.turn()).score(mate_score=2500)
        entry["centipawn_loss"] = max(entry["eval"] - next_analysis["score"].pov(game.turn()).score(mate_score=2500) ,0)
    else:
        entry["eval"] = game.eval().pov(game.turn()).score(mate_score=2500)
        entry["centipawn_loss"] = max(game.eval().pov(game.turn()).score(mate_score=2500) - game.next().eval().pov(game.turn()).score(mate_score=2500) ,0)
    return entry

all_data = []

games_per_player = 100

PGN_DIR = 'new_PGNs/'
game_count = 0
for pgn_batch in tqdm(os.listdir(PGN_DIR)):
    games_processed = 0
    games_tried = 0
    print ('new file', pgn_batch)
    pgn = open(PGN_DIR + pgn_batch, encoding="utf-8")
    while games_processed < games_per_player and games_tried < 100:
        games_tried += 1
        try:
            game = chess.pgn.read_game(pgn)
            if game is None:
                print('game type None, continuing...')
                continue
        except UnicodeDecodeError:
            print('error in parsing game')
            continue
        
        
        white_elo = float(game.headers["WhiteElo"])
        black_elo = float(game.headers["BlackElo"])
        
        try:
            board = game.board()  # set the game board
        except ValueError as e:
            print('variant error', e)
            # some sort of variant issue
            continue
        
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
        while game is not None and game.next().next() is not None and moves_made < max_moves:
            entry = {}
            if game.turn() == chess.WHITE:
                if white_elo <= 2200:
                    game = game.next()
                    moves_made += 1
                    continue

                entry = get_game_statistics(game, elo=white_elo)
                all_data.append(entry)
            else:
                if black_elo <= 2200:
                    game = game.next()
                    moves_made += 1
                    continue
                entry = get_game_statistics(game, elo=black_elo) 
                all_data.append(entry)
            game = game.next()
            moves_made += 1
        games_processed += 1

with open(os.path.join('Databases','all_data.json'), 'w') as fout:
    json.dump(all_data, fout)
