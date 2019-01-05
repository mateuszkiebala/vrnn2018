import os, chess, uuid, argparse, random
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

def boards_vector_triple(prev_board, next_board, diff_board):
    return np.array([board2array(prev_board), board2array(next_board), board2array(diff_board)])


parser = argparse.ArgumentParser()
parser.add_argument('--games', type=int, default=10, help='Number of games played')
args = parser.parse_args()


class PgnGenerator:
    GOOD_RESULT = [1, 0]
    BAD_RESULT = [0, 1]

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
                boards_vec, result = gen_good(board.copy(), move)
                self.result_boards.append(boards_vec)
                self.result_vector.append(result)

                boards_vec, result = gen_bad(board.copy(), move)
                self.result_boards.append(boards_vec)
                self.result_vector.append(result)

                board.push(move)
                if i % 10 == 0:
                    print("Game number: {0}. Move number: {1}".format(game_index, i))
            game_index += 1
            game = pgn.read_game(pgn_file)
        return [self.result_boards, self.result_vector]


def gen_good(board, move):
    prev_board = board.copy()
    board_with_only_start_piece = gen_single_figure_board(move.from_square, prev_board.piece_at(move.from_square))
    board.push(move)
    return [boards_vector_triple(prev_board, board, board_with_only_start_piece), PgnGenerator.GOOD_RESULT]


def gen_bad(board, move):
    prev_board = board.copy()
    start_square = move.from_square
    piece_to_move = prev_board.piece_at(start_square)
    board_with_only_start_piece = gen_single_figure_board(start_square, piece_to_move)
    illegal_move = gen_illegal_move(prev_board, start_square)
    board.set_piece_at(illegal_move.to_square, piece_to_move)
    board.remove_piece_at(illegal_move.from_square)
    return [boards_vector_triple(prev_board, board, board_with_only_start_piece), PgnGenerator.BAD_RESULT]


def gen_single_figure_board(square, piece):
    board = chess.Board(fen=None)
    board.set_piece_at(square, piece)
    return board


def gen_illegal_move(board, square):
    all_moves = set("{0}{1}{2}".format(chess.SQUARE_NAMES[square], f, r) for f in "abcdefgh" for r in range(1, 9))
    square_legal_moves = set(map(lambda move: move.uci(), filter(lambda move: move.from_square == square, board.legal_moves)))
    illegal_moves = list(all_moves - square_legal_moves)
    random.shuffle(illegal_moves)
    return chess.Move.from_uci(illegal_moves[0])

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


def generate_piece_type_result(board, move):
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
