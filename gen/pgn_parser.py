import os, chess, uuid, argparse
import numpy as np
from chess import svg, pgn
from cairosvg import svg2png
from scipy import misc
from IPython.display import SVG
from common.constants import *

np.set_printoptions(threshold=np.nan)

# Saves board to png with provided name and size
def board2png(board, name, size=DEFAULT_IMAGE_SIZE, coordinates=False):
    svg = SVG(chess.svg.board(board=board, size=size, coordinates=coordinates))
    svg2png(svg.data, write_to=name)

# Returns numpy array from given board
def board2array(board):
    unique_filename = '/tmp/'+str(uuid.uuid4())+'.png'
    board2png(board, unique_filename)
    image = misc.imread(unique_filename)
    os.remove(unique_filename)
    return image

def board2vector(board):
    array = np.array(board2array(board))
    return array.flatten()

def boards_vector(prev_board, next_board):
    return np.array([board2array(prev_board), board2array(next_board)])


parser = argparse.ArgumentParser()
parser.add_argument('--games', type=int, default=10, help='Number of games played')
args = parser.parse_args()

class PgnGenerator:
    def __init__(self, pgn_games_path):
        self.pgn_games_path = pgn_games_path
        self.result_boards = []
        self.result_vector = []

    def generate_data(self):
        pgn_file = open(self.pgn_games_path)
        game = pgn.read_game(pgn_file)
        game_index = 0
        while game and game_index < args.games:
            board = game.board()
            for i, move in enumerate(game.mainline_moves()):
                prev_board = board.copy()
                board.push(move)
                self.result_boards.append(boards_vector(prev_board, board))
                self.result_vector.append(generate_single_result(prev_board, move))
                print("Game number: {0}. Move number: {1}. Last result: {2}".format(game_index, i, self.result_vector[-1]))
            game_index += 1
            game = pgn.read_game(pgn_file)
        return [self.result_boards, self.result_vector]


PIECE_MOVES = {
    1: [1, 0, 0, 0, 1],
    2: [0, 0, 0, 1, 4],
    3: [0, 0, 1, 0, 8],
    4: [1, 1, 0, 0, 8],
    5: [1, 1, 1, 0, 8],
    6: [1, 1, 1, 0, 1]
}

PIECE_TYPE = {
    1: [1, 0, 0, 0, 0, 0],
    2: [0, 1, 0, 0, 0, 0],
    3: [0, 0, 1, 0, 0, 0],
    4: [0, 0, 0, 1, 0, 0],
    5: [0, 0, 0, 0, 1, 0],
    6: [0, 0, 0, 0, 0, 1]
}

def generate_single_result(board, move):
    piece = board.piece_at(move.from_square)
    return PIECE_TYPE[piece.piece_type]
    #return [piece.piece_type, int(piece.color)] + PIECE_MOVES[piece.piece_type]

if __name__ == '__main__':
    gen = PgnGenerator("games/games.pgn")
    boards_path, results_path = dataset_path()

    os.makedirs(os.path.dirname(boards_path), exist_ok=True)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    [boards, results] = gen.generate_data()
    np.save(boards_path, np.array(boards))
    np.save(results_path, np.array(results))
