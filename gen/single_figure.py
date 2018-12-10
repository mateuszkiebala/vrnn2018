import os, chess, numpy, copy
from chess import svg
from cairosvg import svg2png
from scipy import misc
from IPython.display import SVG

# Saves board to png with provided name and size
def board2png(board, name, size=400):
    svg = SVG(chess.svg.board(board=board, coordinates=False, size=size))
    svg2png(svg.data, write_to=name)

# Returns numpy array from given board
def board2array(board):
    board2png(board, 'tmp.png')
    image = misc.imread('tmp.png')
    os.remove('tmp.png')
    return image

# Push uci move in board. Does not check if the move is valid.
def push_move(board, uci):
    move = chess.Move.from_uci(uci)
    board.push(move)

board = chess.Board("8/8/8/8/4N3/8/8/8 w - - 0 1")
os.mkdir('moves')
for move in board.legal_moves:
    new_board = board.copy()
    new_board.push(move)
    board2png(new_board, 'moves/'+str(move)+'.png')

# Printing board
numpy.set_printoptions(threshold=numpy.nan)
image = board2array(board)
# print(image)
