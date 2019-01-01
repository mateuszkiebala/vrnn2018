import os, chess, numpy
from chess import svg, pgn
from cairosvg import svg2png
from scipy import misc
from IPython.display import SVG
from common.constants import DEFAULT_IMAGE_SIZE

numpy.set_printoptions(threshold=numpy.nan)


def board2array(board):
    board2png(board, 'tmp.png')
    image = misc.imread('tmp.png')
    os.remove('tmp.png')
    return image

def board2vector(board):
    array = numpy.array(board2array(board))
    return array.flatten()


def board2png(board, filename, size=DEFAULT_IMAGE_SIZE, coordinates=False):
    svg = SVG(chess.svg.board(board=board, size=size, coordinates=coordinates))
    svg2png(svg.data, write_to=filename)


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


if __name__ == '__main__':
    parse_pgn("../games/games.pgn", 'parsed_game/')
