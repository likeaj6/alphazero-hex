#! /usr/bin/env python3
########################################
# CS63: Artificial Intelligence, Final Project
# Spring 2018, Swarthmore College
########################################

import numpy as np
from scipy.ndimage import label
from keras.models import load_model
import sys

_adj = np.ones([3,3], int)
_adj[0,0] = 0
_adj[2,2] = 0

RED   = u"\033[1;31m"
BLUE  = u"\033[1;34m"
RESET = u"\033[0;0m"
CIRCLE = u"\u25CF"

RED_DISK = RED + CIRCLE + RESET
BLUE_DISK = BLUE + CIRCLE + RESET
EMPTY_CELL = u"\u00B7"

RED_BORDER = RED + "-" + RESET
BLUE_BORDER = BLUE + "\\" + RESET

def print_char(i):
    if i > 0:
        return BLUE_DISK
    if i < 0:
        return RED_DISK
    return EMPTY_CELL

class HexGame:

    def __init__(self, size=8):
        self.size = size
        self.turn = 1
        self.board = np.zeros([size, size], int)

        self._moves = None
        self._terminal = None
        self._winner = None
        self._repr = None
        self._hash = None

    def __repr__(self):
        if self._repr is None:
            self._repr = u"\n" + (" " + RED_BORDER)*self.size +"\n"
            for i in range(self.size):
                self._repr += " " * i + BLUE_BORDER + " "
                for j in range(self.size):
                    self._repr += print_char(self.board[i,j]) + " "
                self._repr += BLUE_BORDER + "\n"
            self._repr += " "*(self.size) + " " + (" " + RED_BORDER) * self.size
        return self._repr

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(repr(self))
        return self._hash

    def __eq__(self, other):
        return repr(self) == repr(other)

    def makeMove(self, move):
        """Returns a new ConnectionGame in which move has been played.
        A move is a column into which a piece is dropped."""
        hg = HexGame(self.size)
        hg.board = np.array(self.board)
        hg.board[move[0], move[1]] = self.turn
        hg.turn = -self.turn
        return hg

    @property
    def availableMoves(self):
        if self._moves is None:
            self._moves = list(zip(*np.nonzero(np.logical_not(self.board))))
        return self._moves

    @property
    def isTerminal(self):
        if self._terminal is not None:
            return self._terminal
        if self.turn == 1:
            clumps = label(self.board < 0, _adj)[0]
        else:
            clumps = label(self.board.T > 0, _adj)[0]
        spanning_clumps = np.intersect1d(clumps[0], clumps[-1])
        self._terminal = np.count_nonzero(spanning_clumps)
        return self._terminal

    @property
    def winner(self):
        if self.isTerminal:
            return -self.turn
        return 0


from BasicPlayers import HumanPlayer, RandomPlayer
from HexPlayerBryce import HexPlayerBryce
#from MonteCarloTreeSearch import MCTSPlayer
from AlphaHex import  DeepLearningPlayer

players = {"random":RandomPlayer,
           "human":HumanPlayer,
#           "mcts":MCTSPlayer,
           "bryce":HexPlayerBryce,
           "drl":DeepLearningPlayer}

from argparse import ArgumentParser

def play_game(game, player1, player2, show=False):
    """Plays a game then returns the final state."""
    while not game.isTerminal:
        if show:
            print(game)
        if game.turn == 1:
            m = player1.getMove(game)
        else:
            m = player2.getMove(game)
        if m not in game.availableMoves:
            raise Exception("invalid move: " + str(m))
        game = game.makeMove(m)
    if show:
        print(game, "\n")
    print("player", print_char(game.winner), "(", end='')
    print((player1.name if game.winner == 1 else player2.name)+") wins")
    return game

def playBryce(current_model, num_games=10, num_rollouts_1=400, num_rollouts_2=400, play_first=True, show=True):
    for i in range(num_games):
        print('Game #: ' + str(i))
        g = HexGame(8)
        if i%2:
            player1 = DeepLearningPlayer(current_model, rollouts=num_rollouts_1, save_tree=True, competitive=True)
            player2 = HexPlayerBryce(rollouts=num_rollouts_2)
        else:
            player2 = DeepLearningPlayer(current_model, rollouts=num_rollouts_1, save_tree=True, competitive=True)
            player1 = HexPlayerBryce(rollouts=num_rollouts_2)
        # player2 = DeepLearningPlayer(current_model)
        game = play_game(g, player1, player2, show)

def playSelf(current_model, num_games=10, num_rollouts_1=400, num_rollouts_2=400, play_first=True, show=True):
      for i in range(num_games):
          print('Game #: ' + str(i))
          g = HexGame(8)
          player1 = DeepLearningPlayer(current_model, rollouts=num_rollouts_1, save_tree=True)
          player2 = DeepLearningPlayer(current_model, rollouts=num_rollouts_2, save_tree=True)
          game = play_game(g, player1, player2, show)

def playRandom(current_model, num_games=10, num_rollouts=400, play_first=True, show=True):
    for i in range(num_games):
        print('Game #: ' + str(i))
        g = HexGame(8)
        if play_first:
            player1 = DeepLearningPlayer(current_model, rollouts=num_rollouts_1, save_tree=True, competitive=True)
            player2 = RandomPlayer()
        else:
            player1 = RandomPlayer()
            player2 = DeepLearningPlayer(current_model, rollouts=num_rollouts_1, save_tree=True, competitive=True)
        game = play_game(g, player1, player2, show)

def selfPlay(model_a, model_b, num_games, num_rollouts_1, num_rollouts_2, show):
    wins_a = 0
    for i in range(num_games):
        print('Game #: ' + str(i))
        g = HexGame(8)
        player1 = DeepLearningPlayer(model_a, rollouts=num_rollouts_1, save_tree=True, competitive=True)
        player2 = DeepLearningPlayer(model_b, rollouts=num_rollouts_2, save_tree=True, competitive=True)
        if i%2:
            game = play_game(g, player1, player2, show)
            if(game.winner == 1):
                print('a wins')
                wins_a += 1
            else:
                print('b wins')
        else:
            game = play_game(g, player2, player1, show)
            if(game.winner == -1):
                print('a wins')
                wins_a += 1
            else:
                print('b wins')
    print('model a wins: ' + str(wins_a))

if __name__ == "__main__":
    p = ArgumentParser()
    current_model = load_model('new_supervised_zero.h5')
    # playBryce(current_model, 10, 200, 200, True, False)
    playSelf(current_model, 10, 400, 400, False, show=True)
    playBryce(current_model, 20, 300, 300, True, False)
    # playBryce(current_model, 10, 200, 200, False, True)
    # playRandom(current_model, 10, 400, True, True)
