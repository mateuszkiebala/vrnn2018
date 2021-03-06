import os, chess, uuid, threading
import numpy as np
from chess import svg, pgn
from cairosvg import svg2png
from scipy import misc
from IPython.display import SVG
from common.constants import DEFAULT_IMAGE_SIZE, BOOK_GAMES_PATH

# Saves board to png with provided name and size
def board2png(board, name, size=DEFAULT_IMAGE_SIZE, coordinates=False):
    svg = SVG(chess.svg.board(board=board, size=size, coordinates=coordinates))
    svg2png(svg.data, write_to=name)

def board2posvec(board):
    pieces = [0] * 768 # (6 white pieces + 6 black) * 64 squares

    for square in range(0, 64):
        piece = board.piece_at(square)

        if piece is not None:
            square_offset = 12 * square

            color_offset = 0
            if piece.color == chess.BLACK:
                color_offset = 6

            piece_offset = piece.piece_type + color_offset - 1
            pieces[square_offset + piece_offset] = 1

    return np.array(pieces)

# Returns numpy array from given board
def board2array(board, gray_scale=False):
    unique_filename = '/tmp/'+str(uuid.uuid4())+'.png'
    board2png(board, unique_filename)
    if not gray_scale:
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
        self.cur_pgn_idx = 0
        self.pgns = [pgn for pgn in os.listdir(BOOK_GAMES_PATH)]
        self.pgn = open(os.path.join(BOOK_GAMES_PATH, self.pgns[self.cur_pgn_idx]))

    def _next_pgn(self):
        self.cur_pgn_idx += 1
        self.cur_pgn_idx %= len(self.pgns)
        self.pgn = open(os.path.join(BOOK_GAMES_PATH, self.pgns[self.cur_pgn_idx]))

    def next_book_game(self):
        with self.lock:
            while True:
                try:
                    game = chess.pgn.read_game(self.pgn)
                except:
                    print("Error reading a game from pgn {}" + self.pgns[self.cur_pgn_idx])
                    continue

                if game is not None:
                    print("Next book game from {}".format(self.pgns[self.cur_pgn_idx]))
                    return game
                else:
                    self._next_pgn()

if __name__ == '__main__':
    parse_pgn("../games/games.pgn", 'parsed_game/')
