import os, chess, uuid, threading
import numpy as np
from chess import svg, pgn
from cairosvg import svg2png
from scipy import misc
from IPython.display import SVG
from common.constants import DEFAULT_IMAGE_SIZE, BOOK_GAMES_PATH

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

def pgn2board(pgn_game, directory):
    board = pgn_game.board()
    if not os.path.exists(directory):
        os.makedirs(directory)

    for i, move in enumerate(pgn_game.mainline_moves()):
        board.push(move)
        board2png(board, '{0}/game{1}.png'.format(directory, i))


def parse_pgn(input_path, output_path):
    pgn_file = open(input_path)
    game = pgn.read_game(pgn_file)
    index = 0

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    while game:
        pgn2board(game, '{0}/{1}'.format(output_path, index))
        index += 1

class PgnReader:
    pgn = open(BOOK_GAMES_PATH)
    lock = threading.Lock()
    
def next_book_game():
    with PgnReader.lock:
        while True:
            try:
                print("Reading next_book_game")
                return chess.pgn.read_game(PgnReader.pgn)
            except Exception:
                print("ValueError reading game from pgn")
                continue

if __name__ == '__main__':
    parse_pgn("../games/games.pgn", 'parsed_game/')
