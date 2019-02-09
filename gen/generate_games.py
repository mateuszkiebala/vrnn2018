import numpy as np
import multiprocessing as mp
import os, chess, random, copy, time, argparse
from pgn_parser import board2png, board2array, board2posvec, PgnReader
from common.constants import *

parser = argparse.ArgumentParser()
parser.add_argument('--maxmoves', type=int, default=100, help='Max moves until the game is counted as finished')
parser.add_argument('--randomgames', type=int, default=25, help='Number of random games played')
parser.add_argument('--bookgames', type=int, default=25, help='Number of book games played')
parser.add_argument('--extlabels', action='store_true', help='Determines if generator should generate extended labels')
parser.add_argument('--pngs', action='store_true', help='Determines if boards pngs should be generated')
parser.add_argument('--grayscale', action='store_true', help='Determines if grayscale should be used')
parser.add_argument('--posvec', action='store_true', help='Generate compact position vector instread of images')
args = parser.parse_args()

pgn_reader = PgnReader()

def gen_labels(valid, board, next_move, extended=False):
    if not extended:
        return [int(valid), int(not valid)]

    uci = str(next_move)
    from_pos, to_pos = uci[:2], uci[2:]

    from_square = chess.square(ord(from_pos[0]) - ord('a'), int(from_pos[1])-1)
    to_square = chess.square(ord(to_pos[0]) - ord('a'), int(to_pos[1])-1)

    distance = chess.square_distance(from_square, to_square)
    distance_vec = [0] * 8
    distance_vec[distance-1] = 1

    symbol = board.piece_at(from_square).symbol()
    is_black, is_white = int(symbol.islower()), int(symbol.isupper())

    symbol_vec = [0] * len(PIECE_TO_INT)
    symbol_vec[PIECE_TO_INT[symbol.lower()]] = 1

    labels = []
    labels.extend([int(valid), int(not valid)]) # is valid or not valid
    labels.extend(distance_vec) # what distance
    labels.extend([is_black, is_white]) # is black or white
    labels.extend(symbol_vec) # which symbol
    return labels

def boards_vector(prev_board, next_board, gray_scale=False, posvec=False):
    if posvec:
        return np.array([board2posvec(prev_board), board2posvec(next_board)])

    return np.array([board2array(prev_board, gray_scale), board2array(next_board, gray_scale)])

def end_reason(board, move_number):
    move_prefix = "[" + str(move_number) + "] "
    if move_number > args.maxmoves:
        return move_prefix + "Too many moves made: " + str(move_number)

    if board.is_stalemate():
        return move_prefix + "Stalemate"
    if board.is_fivefold_repetition():
        return move_prefix + "Fivefold repetition"
    if board.is_seventyfive_moves():
        return move_prefix + "75 moves without capture and a pawn push"

    return move_prefix + "Game over"

def gen_non_legal_move(board, legal_moves):
    pseudo_legal_moves = [g for g in board.pseudo_legal_moves]
    random.shuffle(pseudo_legal_moves)

    for move in pseudo_legal_moves:
        if move not in legal_moves:
            return move

    return None

class Generator:
    def __init__(self, game_number, extended_labels=False, save_png=False, gray_scale=False, posvec=False):
        self.game_number = game_number
        self.extended_labels = extended_labels
        self.save_png = save_png
        self.gray_scale = gray_scale
        self.posvec = posvec
        self.result_boards = []
        self.result_vector = []

    def gen_wrong_move(self, board):
        available_squares = set()

        for i in range(1, 7):
            available_squares = available_squares.union(board.pieces(i, board.turn))

        while True:
            random_start_square = chess.SQUARE_NAMES[random.choice(list(available_squares))]
            random_dst_square = chess.SQUARE_NAMES[random.randint(0, 63)]

            move = chess.Move.from_uci(random_start_square + random_dst_square)

            if move not in board.legal_moves:
                return move


    def _create_imgs_dirs(self):
        if self.save_png:
            if not os.path.exists(GOOD_GAMES_IMG_PATH + str(self.game_number)):
                os.makedirs(GOOD_GAMES_IMG_PATH + str(self.game_number))
            if not os.path.exists(BAD_GAMES_IMG_PATH + str(self.game_number)):
                os.makedirs(BAD_GAMES_IMG_PATH + str(self.game_number))

    def save_wrong_move(self, board, move_number):
        move = self.gen_wrong_move(board)

        wrong_board = board.copy()
        wrong_board.push(move)

        labels = gen_labels(False, board, move, extended=self.extended_labels)
        self.result_boards.append(boards_vector(board, wrong_board, gray_scale=self.gray_scale, posvec=self.posvec))
        self.result_vector.append(labels)

        if self.save_png:
            board2png(board, BAD_GAMES_IMG_PATH + str(self.game_number) + "/" + str(move_number) + ".png")
            board2png(wrong_board, BAD_GAMES_IMG_PATH + str(self.game_number) + "/" + str(move_number) + "-wrong.png")

    def save_good_move(self, board, next_board, move, move_number):
        labels = gen_labels(True, board, move, extended=self.extended_labels)
        self.result_boards.append(boards_vector(board, next_board, gray_scale=self.gray_scale, posvec=self.posvec))
        self.result_vector.append(labels)
        if self.save_png:
            board2png(board, GOOD_GAMES_IMG_PATH + str(self.game_number) + "/" + str(move_number) + ".png")
            board2png(next_board, GOOD_GAMES_IMG_PATH + str(self.game_number) + "/" + str(move_number) + "-good.png")


    def book_game(self):
        while True:
            try:
                game = pgn_reader.next_book_game()
                print("After next book game")
            except Exception:
                print("Value error with next_book_game, trying another one")
                continue

            if game is None:
                print("End of chess book reached. Couldn't generate any more games")
                return None

            if len([m for m in game.mainline_moves()]) > 0:
                break

        board = game.board()
        self._create_imgs_dirs()
        move_number = 0

        for move in game.mainline_moves():
            legal_moves = [g for g in board.legal_moves]

            if random.random() < SAVE_WRONG_MOVE_PROBABILITY:
                print("[Book {}] Saving wrong move {}".format(self.game_number, move_number))
                self.save_wrong_move(board, move_number)

            next_board = board.copy()
            next_board.push(move)

            if random.random() < SAVE_LEGAL_MOVE_PROBABILITY:
                print("[Book {}] Saving correct move {}".format(self.game_number, move_number))
                self.save_good_move(board, next_board, move, move_number)

            board = next_board
            move_number = move_number + 1

        print("Book game " + str(self.game_number) + " ends on move " + str(move_number))

    def random_game(self, board=chess.Board(), move_number=0):
        if move_number == 0:
            self._create_imgs_dirs()

        if board.is_game_over() or (move_number > args.maxmoves):
            print(end_reason(board, move_number))
            return

        legal_moves = [g for g in board.legal_moves]

        if random.random() < SAVE_WRONG_MOVE_PROBABILITY:
            print("[Random {}] Saving wrong move {}".format(self.game_number, move_number))
            self.save_wrong_move(board, move_number)

        next_move = random.choice(legal_moves)
        next_board = board.copy()
        next_board.push_uci(str(next_move))

        if random.random() < SAVE_LEGAL_MOVE_PROBABILITY:
            print("[Random {}] Saving correct move {}".format(self.game_number, move_number))
            self.save_good_move(board, next_board, next_move, move_number)

        self.random_game(next_board, move_number + 1)

    def results(self):
        return self.result_boards, self.result_vector

def generate_random_game(i):
    generator = Generator(i, extended_labels=args.extlabels, save_png=args.pngs, gray_scale=args.grayscale, posvec=args.posvec)
    generator.random_game()
    return generator.results()

def generate_book_game(i):
    generator = Generator(i, extended_labels=args.extlabels, save_png=args.pngs, gray_scale=args.grayscale, posvec=args.posvec)
    generator.book_game()
    return generator.results()

def save_dataset(all_boards, all_results, number=None):
    if number is None:
        boards_path, results_path = dataset_path()
    else:
        boards_path, results_path = dataset_number_path(number)

    p = np.random.permutation(len(all_boards))

    os.makedirs(os.path.dirname(boards_path), exist_ok=True)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    np.save(boards_path, np.array(all_boards)[p])
    np.save(results_path, np.array(all_results)[p])

    print('Dataset saved into: ', boards_path, 'and', results_path)


# Generate games in parallel
pool = mp.Pool(processes=mp.cpu_count())
all_boards = []
all_results = []

random_game_number = 1
book_game_number = args.randomgames + 1

current_dataset_number = 0

if os.path.exists(GAMES_ARR_PATH):
    dirs = [int(dir) for dir in os.listdir(GAMES_ARR_PATH) if os.path.isdir(os.path.join(GAMES_ARR_PATH, dir))]

    if len(dirs) != 0:
        current_dataset_number = max(dirs) + 1

max_dataset_size = MAX_DATASET_SIZE

if args.posvec:
    max_dataset_size = MAX_POSVEC_DATASET_SIZE

while True:
    random_games_upper_bound = min(args.randomgames+1, random_game_number + mp.cpu_count()) # either next batch of games or smaller if games run out

    for result in pool.map(generate_random_game, range(random_game_number, random_games_upper_bound)):
        all_boards.extend(result[0])
        all_results.extend(result[1])

    random_game_number += mp.cpu_count()

    book_games_upper_bound = min(args.bookgames + args.randomgames + 1, book_game_number + mp.cpu_count())

    for result in pool.map(generate_book_game, range(book_game_number, book_games_upper_bound)):
        all_boards.extend(result[0])
        all_results.extend(result[1])

    book_game_number += mp.cpu_count()

    if len(all_boards) > max_dataset_size:
        save_dataset(all_boards, all_results, current_dataset_number)
        all_boards = []
        all_results = []
        current_dataset_number += 1

    if random_game_number > args.randomgames and book_game_number > (args.randomgames + args.bookgames):
        if len(all_boards) > 0:
            save_dataset(all_boards, all_results, current_dataset_number)
        break


