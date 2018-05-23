########################################
# CS63: Artificial Intelligence, Final Project
# Spring 2018, Swarthmore College
########################################

from random import choice
from sys import stdin

class HumanPlayer:
    """Player that gets moves from command line input."""
    def __init__(self, *args):
        self.name = "Human"

    def getMove(self, game):
        move = None
        while move not in game.availableMoves:
            print("select a row and column")
            try:
                line = stdin.readline().split()
                move = (int(line[0]), int(line[1]))
            except ValueError:
                print("invalid move")
            if move not in game.availableMoves:
                print("invalid move")
        return move

class RandomPlayer:
    """Player that selects a random legal move."""
    def __init__(self, *args):
        self.name = "Random"

    def getMove(self, game):
        return choice(game.availableMoves)
