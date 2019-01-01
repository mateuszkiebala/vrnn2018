import os, chess, numpy, copy
from pgn_parser import board2png, board2array

# Push uci move in board. Does not check if the move is valid.
def push_move(board, uci):
    move = chess.Move.from_uci(uci)
    board.push(move)

board = chess.Board("8/8/8/8/4N3/8/8/8 w - - 0 1")
os.mkdir('moves')
for move in board.legal_moves:
    new_board = chess.Board(board.copy())
    new_board.push(move)
    board2png(new_board, 'moves/'+str(move)+'.png')

# Printing board
numpy.set_printoptions(threshold=numpy.nan)
image = board2array(board)
# print(image)
