import os, chess, random, copy, time

import numpy as np

from pgn_parser import board2png, board2array, board2vector

from chess import svg
from IPython.display import SVG
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF, renderPM

from cairosvg import svg2png

GOOD_GAMES_IMG_PATH = "../games/images/good/"
BAD_GAMES_IMG_PATH = "../games/images/bad/"

GAMES_ARR_PATH = "../games/dataset/"

SAMPLE_SIZE = 1
BAD_SAMPLE_SIZE = 1

MAX_MOVE_NUM = 1000
GAMES_NUM = 50

result_boards = []
result_vector = []

def boards_vector(prev_board, next_board):
    return np.concatenate([board2vector(prev_board), board2vector(next_board)])

def end_reason(board, move_number):
    move_prefix = "[" + str(move_number) + "] "
    if move_number > MAX_MOVE_NUM:
        return move_prefix + "Too many moves made: " + str(move_number)

    if board.is_stalemate():
        return move_prefix + "Stalemate"
    if board.is_fivefold_repetition():
        return move_prefix + "Fivefold repetition"
    if board.is_seventyfive_moves():
        return move_prefix + "75 moves without capture and a pawn push"

    return move_prefix + "Game over"

def gen_non_legal_moves(board, legal_moves):
    pseudo_legal_moves = set([g for g in board.pseudo_legal_moves])

    non_legal_moves = set(pseudo_legal_moves)
    non_legal_moves.difference_update(set(legal_moves))
    return non_legal_moves

def generate_random_game(board, game_number, move_number):
    global resultBoards, resultVector

    if move_number == 0:
        if not os.path.exists(GOOD_GAMES_IMG_PATH + str(game_number)):
            os.makedirs(GOOD_GAMES_IMG_PATH + str(game_number))
        if not os.path.exists(BAD_GAMES_IMG_PATH + str(game_number)):
            os.makedirs(BAD_GAMES_IMG_PATH + str(game_number))

    # save move to png
    board2png(board, GOOD_GAMES_IMG_PATH + str(game_number) + "/" + str(move_number) + ".png")

    if move_number % 10 == 0:
        print("Game: " + str(game_number) + ", Move: " + str(move_number))

    if board.is_game_over():
        print(end_reason(board, move_number))
        return

    legal_moves = [g for g in board.legal_moves]
    non_legal_moves = gen_non_legal_moves(board, set(legal_moves))

    if len(non_legal_moves) > 0:
        non_legal_board = board.copy()
        non_legal_board.push(non_legal_moves.pop())

        result_boards.append(boards_vector(board, non_legal_board))
        result_vector.append(0)

        board2png(board, BAD_GAMES_IMG_PATH + str(game_number) + "/" + str(move_number) + ".png")
        board2png(non_legal_board, BAD_GAMES_IMG_PATH + str(game_number) + "/" + str(move_number) + "-bad.png")

    next_move = random.choice(legal_moves)
    next_board = board.copy()
    next_board.push_uci(str(next_move))

    result_boards.append(boards_vector(board, next_board))
    result_vector.append(1)

    generate_random_game(next_board, game_number, move_number + 1)
    return


for i in range(GAMES_NUM):
    board = chess.Board()
    result_boards = []
    result_vector = []
    generate_random_game(board, i, 0)

    if not os.path.exists(GAMES_ARR_PATH + str(i)):
        os.makedirs(GAMES_ARR_PATH + str(i))

    np.save(GAMES_ARR_PATH + str(i) + "/boards.npy", np.array(result_boards))
    np.save(GAMES_ARR_PATH + str(i) + "/results.npy", np.array(result_vector))



