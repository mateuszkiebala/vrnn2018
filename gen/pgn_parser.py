import os, chess, uuid, threading
import numpy as np
from chess import svg, pgn
from cairosvg import svg2png
from scipy import misc
from IPython.display import SVG
from common.constants import DEFAULT_IMAGE_SIZE, BOOK_GAMES_PATH, BOOK_GAMES_PATH

np.set_printoptions(threshold=np.nan)

# Saves board to png with provided name and size
def board2png(board, name, size=DEFAULT_IMAGE_SIZE, coordinates=False):
    svg = SVG(chess.svg.board(board=board, size=size, coordinates=coordinates))
    svg2png(svg.data, write_to=name)

def board2posvec(board):
    pieces = []

    for square in range(0, 63):
        piece = board.piece_at(square)
        piece_type = piece.piece_type
        piece_color = piece.piece_color

        if piece_color == chess.WHITE:
            col_mul = 1
        else:
            col_mul = 2

        pieces.append(piece_type * col_mul)

    print("board2posvec res {}".format(pieces))
    
    return np.array(pieces)

# Returns numpy array from given board
def board2array(board, gray_sclale=False, ):
    unique_filename = '/tmp/'+str(uuid.uuid4())+'.png'
    board2png(board, unique_filename)
    if not gray_sclale:
        image = misc.imread(unique_filename)
        os.remove(unique_filename)
        return image

    image = misc.imread(unique_filename, mode='F')
    os.remove(unique_filename)
    return np.expand_dims(image, axis=2)

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
    def __init__(self):
        self.lock = threading.Lock()
        self.pgns = [pgn for pgn in os.listdir(BOOK_GAMES_PATH)]
        self.current_pgn_name = self.pgns[-1]
        self.pgn = open(os.path.join(BOOK_GAMES_PATH, self.pgns.pop()))

    def _next_pgn(self):
        if len(self.pgns) == 0:
            self.pgns = [pgn for pgn in os.listdir(BOOK_GAMES_PATH)]

        self.current_pgn_name = self.pgns[-1]
        self.pgn = open(os.path.join(BOOK_GAMES_PATH, self.pgns.pop()))


    def next_book_game(self):
        with self.lock:
            while True:
                try:
                    game = chess.pgn.read_game(self.pgn)
                except:
                    print("Error reading a game from pgn {}" + self.current_pgn_name)
                    continue

                if game is not None:
                    print("Next book game from {}".format(self.current_pgn_name))
                    return game

                self._next_pgn()

if __name__ == '__main__':
    parse_pgn("../games/games.pgn", 'parsed_game/')
