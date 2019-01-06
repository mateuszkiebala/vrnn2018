import os, chess, uuid, argparse, random, threading
import multiprocessing as mp
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


def boards_vector_double(prev_board, next_board):
    return np.array([board2array(prev_board), board2array(next_board)])


parser = argparse.ArgumentParser()
parser.add_argument('--games', type=int, default=10, help='Number of games played')
parser.add_argument('--result_type', choices=['bad', 'good', 'piece_type', 'move_length'],  required=True, help='Type of result')
args = parser.parse_args()

GOOD_RESULT = [1, 0]
BAD_RESULT = [0, 1]


class PgnReader:
    pgn = open(BOOK_GAMES_PATH)
    lock = threading.Lock()


def next_book_game():
    game = None
    while not game:
        try:
            with PgnReader.lock:
                game = chess.pgn.read_game(PgnReader.pgn)
        except ValueError:
            print("Value error with next_book_game, trying another one")
            continue
    return game


def generate_data(game_num):
    if args.result_type == 'move_length':
        generate_fun = generate_move_length_data
    elif args.result_type == 'piece_type':
        generate_fun = generate_piece_type_data
    elif args.result_type == 'good':
        generate_fun = generate_good
    else:
        generate_fun = generate_bad

    result_boards = []
    result_vector = []
    game = next_book_game()
    board = game.board()
    moves_cnt = 0
    for move in game.mainline_moves():
        _board, _result = generate_fun(board.copy(), move)
        result_boards.append(_board)
        result_vector.append(_result)
        board.push(move)
        if moves_cnt % 10 == 0:
            print("Game: {0}. Move number: {1}".format(game_num, moves_cnt))
        moves_cnt += 1
    print("Game: {0} finished with {1} moves".format(game_num, moves_cnt))
    return [result_boards, result_vector]


def generate_move_length_data(board, move):
    piece_to_move = board.piece_at(move.from_square)
    board_single_piece_before = gen_single_figure_board(move.from_square, piece_to_move)
    board_single_piece_after = gen_single_figure_board(move.to_square, piece_to_move)
    return [boards_vector_double(board_single_piece_before, board_single_piece_after), gen_move_length_result(move)]


def generate_piece_type_data(board, move):
    board_single_piece = gen_single_figure_board(move.from_square, board.piece_at(move.from_square))
    return [board2array(board_single_piece), gen_piece_type_result(board, move)]


def generate_good(board, move):
    prev_board = board.copy()
    board.push(move)
    return [boards_vector_double(prev_board, board), GOOD_RESULT]


def generate_bad(board, move):
    prev_board = board.copy()
    start_square = move.from_square
    piece_to_move = prev_board.piece_at(start_square)
    illegal_move = gen_illegal_move(prev_board, start_square)
    board.set_piece_at(illegal_move.to_square, piece_to_move)
    board.remove_piece_at(illegal_move.from_square)
    return [boards_vector_double(prev_board, board), BAD_RESULT]


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


def gen_piece_type_result(board, move):
    result_vec = [0] * 6
    result_vec[board.piece_at(move.from_square).piece_type - 1] = 1
    return result_vec


def gen_move_length_result(move):
    result_vec = [0] * 7
    result_vec[chess.square_distance(move.from_square, move.to_square) - 1] = 1
    return result_vec

if __name__ == '__main__':
    pool = mp.Pool(processes=mp.cpu_count())
    all_boards = []
    all_results = []
    for result in pool.map(generate_data, range(1, args.games + 1)):
        all_boards.extend(result[0])
        all_results.extend(result[1])

    boards_path, results_path = dataset_path()

    os.makedirs(os.path.dirname(boards_path), exist_ok=True)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    np.save(boards_path, np.array(all_boards))
    np.save(results_path, np.array(all_results))
