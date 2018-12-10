import os, chess, random, copy, time

import numpy as np

from pgn_parser import board2png, board2array, board2vector

from chess import svg
from IPython.display import SVG
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF, renderPM

from cairosvg import svg2png

GOOD_GAMES_IMG_PATH = "../games/images/good/"
BAD_GAMES_IMG_PATH = "../games/images/bad/"

GAMES_ARR_PATH = "../games/arrays/"


SAMPLE_SIZE = 1
BAD_SAMPLE_SIZE = 1

MAX_MOVE_NUM = 1000
GAMES_NUM = 50

resultBoards = []
resultVector = []

def getBoardsVector(prevBoard, nextBoard):
    return np.concatenate([board2vector(prevBoard), board2vector(nextBoard)])

def getEndReason(board, moveNumber):
    movePrefix = "[" + str(moveNumber) + "] "
    if moveNumber > MAX_MOVE_NUM:
        return movePrefix + "Too many moves made: " + str(moveNumber)

    if board.is_stalemate():
        return movePrefix + "Stalemate"
    if board.is_fivefold_repetition():
        return movePrefix + "Fivefold repetition"
    if board.is_seventyfive_moves():
        return movePrefix + "75 moves wihout capture and a pawn push"
    
    return movePrefix + "Game over"

def getNonLegalMoves(board, legalMoves):
    pseudoLegalMoves = set([g for g in board.pseudo_legal_moves])

    notLegalMoves = set(pseudoLegalMoves)
    notLegalMoves.difference_update(set(legalMoves))

    return notLegalMoves

def generateRandomGame(board, gameNumber, moveNumber):
    global resultBoards, resultVector

    if moveNumber == 0:
        if not os.path.exists(GOOD_GAMES_IMG_PATH + str(gameNumber)):
            os.makedirs(GOOD_GAMES_IMG_PATH + str(gameNumber))
        if not os.path.exists(BAD_GAMES_IMG_PATH + str(gameNumber)): 
            os.makedirs(BAD_GAMES_IMG_PATH + str(gameNumber))

    # save move to png
    board2png(board, GOOD_GAMES_IMG_PATH + str(gameNumber) + "/" + str(moveNumber) + ".png")

    if moveNumber % 10 == 0:
        print("Game: " + str(gameNumber) + ", Move: " + str(moveNumber))

    if board.is_game_over():
        print(getEndReason(board, moveNumber))
        return

    legalMoves = [g for g in board.legal_moves]
    notLegalMoves = getNonLegalMoves(board, set(legalMoves))

    if len(notLegalMoves) > 0:
        nonLegalBoard = board.copy()
        nonLegalBoard.push(notLegalMoves.pop())

        resultBoards.append(getBoardsVector(board, nonLegalBoard))
        resultVector.append(0)

        board2png(board, BAD_GAMES_IMG_PATH + str(gameNumber) + "/" + str(moveNumber) + ".png")
        board2png(nonLegalBoard, BAD_GAMES_IMG_PATH + str(gameNumber) + "/" + str(moveNumber) + "-bad.png")


    nextMove = random.choice(legalMoves)
    nextBoard = board.copy()
    nextBoard.push_uci(str(nextMove))

    resultBoards.append(getBoardsVector(board, nextBoard))
    resultVector.append(1)

    generateRandomGame(nextBoard, gameNumber, moveNumber + 1)




for i in range(GAMES_NUM):
    board = chess.Board()
    resultBoards = []
    resultVector = []
    generateRandomGame(board, i, 0)

    if not os.path.exists(GAMES_ARR_PATH + str(i)):
        os.makedirs(GAMES_ARR_PATH + str(i))

    np.save(GAMES_ARR_PATH + str(i) + "/boards.npy", np.array(resultBoards))
    np.save(GAMES_ARR_PATH + str(i) + "/results.npy", np.array(resultVector))



