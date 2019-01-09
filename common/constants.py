from os import path

file_path = path.dirname(path.abspath(__file__))

GOOD_GAMES_IMG_PATH = path.join(file_path, "../games/images/good/")
BAD_GAMES_IMG_PATH = path.join(file_path, "../games/images/bad/")
BOOK_GAMES_PATH = path.join(file_path, "../common/games_pgns/")

GAMES_ARR_PATH = path.join(file_path, "../dataset/")
BOARDS_FILE_NAME = "boards.npy"
RESULTS_FILE_NAME = "results.npy"

SINGLE_MODEL_NAME = "single_model.h5"

DEFAULT_IMAGE_SIZE = 200 # size (width, height) of an image

MAX_DATASET_SIZE = 2000 # max number of boards saved in single .npy file

SAVE_LEGAL_MOVE_PROBABILITY = 0.1
SAVE_ILLEGAL_MOVE_PROBABILITY = 0.03
SAVE_WRONG_MOVE_PROBABILITY = 0.07

def dataset_path():
    return path.join(GAMES_ARR_PATH, BOARDS_FILE_NAME), path.join(GAMES_ARR_PATH, RESULTS_FILE_NAME)

def dataset_number_path(number):
    return path.join(GAMES_ARR_PATH, str(number), BOARDS_FILE_NAME), path.join(GAMES_ARR_PATH, str(number), RESULTS_FILE_NAME)
