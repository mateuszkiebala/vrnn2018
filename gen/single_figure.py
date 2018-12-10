import os, chess, numpy
from chess import svg
from cairosvg import svg2png
from scipy import misc
from IPython.display import SVG

board = chess.Board("8/8/8/8/4N3/8/8/8 w - - 0 1")
svg = SVG(chess.svg.board(board=board))
svg2png(svg.data, write_to='tmp.png')
image = misc.imread('tmp.png')
os.remove('tmp.png')
numpy.set_printoptions(threshold=numpy.nan)
