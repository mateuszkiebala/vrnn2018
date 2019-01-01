import numpy as np
import multiprocessing as mp
import os, chess, random, copy, time, argparse
from pgn_parser import board2png, board2array
from common.constants import *

parser = argparse.ArgumentParser()
parser.add_argument('--maxmoves', type=int, default=100, help='Max moves until the game is counted as finished')
parser.add_argument('--games', type=int, default=50, help='Number of games played')
parser.add_argument('--pngs', type=bool, default=False, help='Determines if boards pngs should be generated')
args = parser.parse_args()

def boards_vector(prev_board, next_board):
    return np.array([board2array(prev_board), board2array(next_board)])

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

class Generator:
    def __init__(self, game_number, save_png=False):
        self.game_number = game_number
        self.save_png = save_png
        self.result_boards = []
        self.result_vector = []

    def random_game(self, board=chess.Board(), move_number=0):
        if move_number == 0 and self.save_png:
            if not os.path.exists(GOOD_GAMES_IMG_PATH + str(self.game_number)):
                os.makedirs(GOOD_GAMES_IMG_PATH + str(self.game_number))
            if not os.path.exists(BAD_GAMES_IMG_PATH + str(self.game_number)):
                os.makedirs(BAD_GAMES_IMG_PATH + str(self.game_number))

        # save move to png
        if self.save_png:
            board2png(board, GOOD_GAMES_IMG_PATH + str(self.game_number) + "/" + str(move_number) + ".png")

        if move_number % 10 == 0:
            print("Game: " + str(self.game_number) + ", Move: " + str(move_number))

        if board.is_game_over() or (move_number > args.maxmoves):
            print(end_reason(board, move_number))
            return

        legal_moves = [g for g in board.legal_moves]
        non_legal_moves = gen_non_legal_moves(board, set(legal_moves))

        if len(non_legal_moves) > 0:
            non_legal_board = board.copy()
            non_legal_board.push(non_legal_moves.pop())

            self.result_boards.append(boards_vector(board, non_legal_board))
            self.result_vector.append([0, 1])

            if self.save_png:
                board2png(board, BAD_GAMES_IMG_PATH + str(self.game_number) + "/" + str(move_number) + ".png")
                board2png(non_legal_board, BAD_GAMES_IMG_PATH + str(self.game_number) + "/" + str(move_number) + "-bad.png")

        next_move = random.choice(legal_moves)
        next_board = board.copy()
        next_board.push_uci(str(next_move))

        self.result_boards.append(boards_vector(board, next_board))
        self.result_vector.append([1, 0])

        self.random_game(next_board, move_number + 1)
    
    def results(self):
        return self.result_boards, self.result_vector

def generate_game(i):
    generator = Generator(i, save_png=args.pngs)
    generator.random_game()
    return generator.results()


# Generate games in parallel
pool = mp.Pool(processes=mp.cpu_count())
all_boards = []
all_results = []
for result in pool.map(generate_game, range(1, args.games+1)):
    all_boards.extend(result[0])
    all_results.extend(result[1])

# Save dataset
boards_path, results_path = dataset_path()

os.makedirs(os.path.dirname(boards_path), exist_ok=True)
os.makedirs(os.path.dirname(results_path), exist_ok=True)

np.save(boards_path, np.array(all_boards))
np.save(results_path, np.array(all_results))

print('Dataset saved into: ', boards_path, 'and', results_path)


