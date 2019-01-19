from os import path

file_path = path.dirname(path.abspath(__file__))

GOOD_GAMES_IMG_PATH = path.join(file_path, "../games/images/good/")
BAD_GAMES_IMG_PATH = path.join(file_path, "../games/images/bad/")
BOOK_GAMES_PATH = path.join(file_path, "../common/games_pgns/")

GAMES_ARR_PATH = path.join(file_path, "../dataset/")
BOARDS_FILE_NAME = "boards.npy"
RESULTS_FILE_NAME = "results.npy"

SINGLE_MODEL_NAME = "single_model.h5"
DUAL_MODEL_NAME = "dual_model.h5"

DEFAULT_IMAGE_SIZE = 200 # size (width, height) of an image
POSVEC_SIZE = 768 # 12 * 64

MAX_DATASET_SIZE = 2000 # max number of boards saved in single .npy file
MAX_POSVEC_DATASET_SIZE = 50000

SAVE_LEGAL_MOVE_PROBABILITY = 0.1
SAVE_ILLEGAL_MOVE_PROBABILITY = 0.03
SAVE_WRONG_MOVE_PROBABILITY = 0.07

EPOCHS_BATCH = 10

PIECE_TO_INT = {"p": 0, "n": 1, "b": 2, "r": 3, "q": 4, "k": 5}

def dataset_path():
    return path.join(GAMES_ARR_PATH, BOARDS_FILE_NAME), path.join(GAMES_ARR_PATH, RESULTS_FILE_NAME)

def dataset_number_path(number):
    return path.join(GAMES_ARR_PATH, str(number), BOARDS_FILE_NAME), path.join(GAMES_ARR_PATH, str(number), RESULTS_FILE_NAME)
