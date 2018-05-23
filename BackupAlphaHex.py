from math import log, sqrt
from numpy.random import choice
from numpy import array
import time

class Node(object):
    """Node used in MCTS"""
    def __init__(self, state, parent_node, prior_prob):
        self.state = state
        self.children = {} # maps moves to Nodes
        self.visits = 0
        self.value = 0
        self.prior_prob = prior_prob
        self.parent_node = parent_node

    def updateValue(self, outcome):
        """Updates the value estimate for the node's state."""
        self.value = (self.visits*self.value + outcome + 1)/(self.visits+1)
        self.visits += 1

    def UCBWeight(self, parent_visits, UCB_const, player):
        """Weight from the UCB formula used by parent to select a child."""
        if player == -1:
            return (1-self.value) + UCB_const*(1-self.prior_prob)*sqrt(parent_visits)/(1+self.visits)
        else:
            # print('is player 1')
            # print('state: ', self.state)
            # print('value: ', self.value)
            # print('prob: ', self.prior_prob)
            # print('visits: ', self.visits)
            return self.value + UCB_const*self.prior_prob*sqrt(parent_visits)/(1+self.visits)

class MCTS:
    def __init__(self, model, player, UCB_const=0.5):
        self.visited_nodes = {} # maps state to node
        self.player = player
        self.model = model
        self.UCB_const = UCB_const

    def runSearch(self, root_node, num_searches):
        # start search from root
        for i in range(num_searches):
            selected_node = root_node
            available_moves = selected_node.state.availableMoves
            # if we've already explored this node, continue down path until we reach a node we haven't expanded yet by selecting node w/ largest UCB weight
            while len(available_moves) == len(selected_node.children) and not selected_node.state.isTerminal:
                # print('already expanded node')
                # select node that maximizes Upper Confidence Bound
                selected_node = self._select(selected_node)
                available_moves = selected_node.state.availableMoves
            if selected_node.state not in self.visited_nodes:
                # print('expanding new node')
                self._expand(selected_node)
                self.visited_nodes[selected_node.state] = selected_node
            self._backprop(selected_node, root_node)
        # return


    def _select(self, parent_node):
        '''returns node with max UCB Weight'''
        children = parent_node.children
        items = children.items()
        UCB_weights = [(v.UCBWeight(parent_node.visits, self.UCB_const, parent_node.state.turn), v) for k,v in items]
        # choose the action with max UCB
        node = max(UCB_weights, key=lambda c: c[0])
        return node[1]

    def _expand(self, node):
        if node.state.turn == -1:
            board = (-1*node.state.board).T.reshape((1, 1, 8, 8))
        else:
            board = node.state.board.reshape((1, 1, 8, 8))
        # start_time = time.time()
        probs, value = self.model.predict(board)
        # print("Finished in : --- %s seconds ---" % (time.time() - start_time))
        if node.state.turn == -1:
            probs = probs.T
        probs = probs.reshape((8, 8))
        # print(probs)
        for move in node.state.availableMoves:
            prob = probs[move[0]][move[1]]  
            if node.state.turn == -1:
                prob = probs[move[1]][move[0]]
            node.children[move] = Node(node.state.makeMove(move), node, prob)
        node.value = value[0][0]
        # print(value)
        # print(node.value[0])
        return probs, value

    def _backprop(self, selected_node, root_node):
        current_node = selected_node
        outcome = selected_node.value
        # print(outcome)
        if selected_node.state.isTerminal:
            outcome = 1 if selected_node.state.winner == 1 else 0
        while current_node != root_node:
            current_node.updateValue(outcome)
            current_node = current_node.parent_node
            # print(current_node.visits)
        # update root node
        root_node.updateValue(outcome)

    def getSearchProbabilities(self, root_node):
        children = root_node.children
        items = children.items()
        child_visits = [child.visits for action, child in items]
        sum_visits = sum(child_visits)
        if sum_visits != 0:
            normalized_probs = {action: (child.visits/sum_visits) for action, child in items}
        else:
            normalized_probs = {action: (child.visits/len(child_visits)) for action, child in items}
        return normalized_probs

class DeepLearningPlayer:

    def __init__(self, model, player, rollouts=1600):
        self.name = "AlphaHex" + str(player)
        self.bestModel = model
        self.rollouts = rollouts
        self.player = player
        self.MCTS = None

    def getMove(self, game):
        # if self.MCTS == None:
        self.MCTS = MCTS(self.bestModel, self.player)
        # if game in self.MCTS.visited_nodes:
        #    root_node = self.MCTS.visited_nodes[game]
        # else:
        root_node = Node(game, None, [])
        self.MCTS.runSearch(root_node, self.rollouts)
        searchProbabilities = self.MCTS.getSearchProbabilities(root_node)
        # moves = list(searchProbabilities.keys())
        probs = list(searchProbabilities.values())
        prob_items = searchProbabilities.items()
        # print(moves)
        print(probs)
        # if self-play, choose stochastically
        # chosen_idx = choice(len(moves), p=probs)
        # else if competitive play, choose highest prob move
        best_move = max(prob_items, key=lambda c: c[1])
        # return moves[chosen_idx]
        return best_move[0]
