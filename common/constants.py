from os import path

file_path = path.dirname(path.abspath(__file__))

GOOD_GAMES_IMG_PATH = path.join(file_path, "../games/images/good/")
BAD_GAMES_IMG_PATH = path.join(file_path, "../games/images/bad/")

GAMES_ARR_PATH = path.join(file_path, "../dataset/")
BOARDS_FILE_NAME = "boards.npy"
RESULTS_FILE_NAME = "results.npy"

DEFAULT_IMAGE_SIZE = 200 # size (width, height) of an image

def dataset_path():
    return path.join(GAMES_ARR_PATH, BOARDS_FILE_NAME), path.join(GAMES_ARR_PATH, RESULTS_FILE_NAME)
