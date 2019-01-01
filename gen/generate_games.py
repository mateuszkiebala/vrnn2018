import numpy as np
import os, chess, random, copy, time, argparse
from gen.generate_games import board2png, board2array, board2vector
from common.constants import *

parser = argparse.ArgumentParser()
parser.add_argument('--maxmoves', type=int, default=100, help='Max moves until the game is counted as finished')
parser.add_argument('--games', type=int, default=50, help='Number of games played')
parser.add_argument('--pngs', type=bool, default=False, help='Determines if boards pngs should be generated')
args = parser.parse_args()

result_boards = []
result_vector = []

def boards_vector(prev_board, next_board):
    return np.concatenate([board2vector(prev_board), board2vector(next_board)])

def end_reason(board, move_number):
    move_prefix = "[" + str(move_number) + "] "
    if move_number > args.maxmoves:
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

def generate_random_game(board, game_number, move_number, save_png=False):
    global resultBoards, resultVector

    if move_number == 0 and save_png:
        if not os.path.exists(GOOD_GAMES_IMG_PATH + str(game_number)):
            os.makedirs(GOOD_GAMES_IMG_PATH + str(game_number))
        if not os.path.exists(BAD_GAMES_IMG_PATH + str(game_number)):
            os.makedirs(BAD_GAMES_IMG_PATH + str(game_number))

    # save move to png
    if save_png:
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

        if save_png:
            board2png(board, BAD_GAMES_IMG_PATH + str(game_number) + "/" + str(move_number) + ".png")
            board2png(non_legal_board, BAD_GAMES_IMG_PATH + str(game_number) + "/" + str(move_number) + "-bad.png")

    next_move = random.choice(legal_moves)
    next_board = board.copy()
    next_board.push_uci(str(next_move))

    result_boards.append(boards_vector(board, next_board))
    result_vector.append(1)

    generate_random_game(next_board, game_number, move_number + 1, save_png=save_png)
    return

all_boards = []
all_results = []
for i in range(args.games):
    board = chess.Board()
    result_boards = []
    result_vector = []
    generate_random_game(board, i, 0, save_png=args.pngs)

    all_boards.extend(result_boards)
    all_results.extend(result_vector)

boards_path, results_path = dataset_path()
np.save(boards_path, np.array(all_boards))
np.save(results_path, np.array(all_results))


