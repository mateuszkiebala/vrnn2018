import os, chess, numpy
from chess import svg
from cairosvg import svg2png
from scipy import misc
from IPython.display import SVG

numpy.set_printoptions(threshold=numpy.nan)

def board2array(board):
    svg = SVG(chess.svg.board(board=board))
    svg2png(svg.data, write_to='tmp.png')
    image = misc.imread('tmp.png')
    os.remove('tmp.png')
    return image

board = chess.Board("8/8/8/8/4N3/8/8/8 w - - 0 1")
image = board2array(board)
print(image)
