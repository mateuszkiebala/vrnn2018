import chess.pgn

pgn = open("games/game_1.pgn")
first_game = chess.pgn.read_game(pgn)
second_game = chess.pgn.read_game(pgn)

board = first_game.board()
for move in first_game.mainline_moves():
    board.push(move)

print(board)
