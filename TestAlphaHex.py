from math import log, sqrt
from numpy.random import choice
from numpy import array
import numpy as np

class Node(object):
    """Node used in MCTS"""
    def __init__(self, state, parent_node, prior_prob):
        self.state = state
        self.children = {} # maps moves to Nodes
        self.visits = 0
        self.value = 0
        self.prior_prob = prior_prob
        self.prior_policy = np.zeros((8, 8))
        self.parent_node = parent_node

    def updateValue(self, outcome):
        """Updates the value estimate for the node's state."""
        self.value = (self.visits*self.value + outcome)/(self.visits+1)
        self.visits += 1
    def UCBWeight_noPolicy(self, parent_visits, UCB_const, player):
        if player == -1:
            return (1-self.value) + UCB_const*sqrt(parent_visits)/(1+self.visits)
        else:
            return self.value + UCB_const*sqrt(parent_visits)/(1+self.visits)
    def UCBWeight(self, parent_visits, UCB_const, player):
        """Weight from the UCB formula used by parent to select a child."""
        if player == -1:
            return (1-self.value) + UCB_const*self.prior_prob*sqrt(parent_visits)/(1+self.visits)
        else:
            return self.value + UCB_const*self.prior_prob*sqrt(parent_visits)/(1+self.visits)

class MCTS:
    def __init__(self, model, UCB_const=1, use_policy=True, use_value=True):
        self.visited_nodes = {} # maps state to node
        self.model = model
        self.UCB_const = UCB_const
        self.use_policy = use_policy
        self.use_value = use_value

    def runSearch(self, root_node, num_searches):
        # start search from root
        for i in range(num_searches):
            selected_node = root_node
            print(selected_node.children)
            available_moves = selected_node.state.availableMoves
            # if we've already explored this node, continue down path until we reach a node we haven't expanded yet by selecting node w/ largest UCB weight
            while len(available_moves) == len(selected_node.children) and not selected_node.state.isTerminal:
                # select node that maximizes Upper Confidence Bound
                selected_node = self._select(selected_node)
                available_moves = selected_node.state.availableMoves
            if not selected_node.state.isTerminal:
                if self.use_policy:
                # expansion
                    actual_policy = []
                    for move in selected_node.state.availableMoves:
                        actual_policy.append(selected_node.prior_policy[move])
                        next_state = selected_node.state.makeMove(move)
                        child_node = self.expand(next_state, selected_node.prior_policy[move], selected_node)
                        selected_node.children[move] = child_node
                        outcome = child_node.value
                        selected_node = child_node
                        self._backprop(selected_node, root_node, outcome)
                    probs, value = self.modelPredict(next_state)
                    move = selected_node.state.availableMoves[actual_policy.index(max(actual_policy))]
                else:
                    moves = selected_node.state.availableMoves
                    np.random.shuffle(moves)
                    for move in moves:
                        if not selected_node.state.makeMove(move) in self.nodes:
                            break
            else:
                outcome = 1 if selected_node.state.winner == 1 else 0
                self._backprop(selected_node, root_node, outcome)

    def modelPredict(self, state):
        if state.turn == -1:
            board = (-1*state.board).T.reshape((1, 1, 8, 8))
        else:
            board = state.board.reshape((1, 1, 8, 8))
        if self.use_policy or self.use_value:
            probs, value = self.model.predict(board)
            value = value[0][0]
            probs = probs.reshape((8, 8))
            if state.turn == -1:
                probs = probs.T
        return probs, value
    def expand(self, state, prior_prob, parent):
        child_node = Node(state, parent, prior_prob)
        if child_node.state.turn == -1:
            board = (-1*child_node.state.board).T.reshape((1, 1, 8, 8))
        else:
            board = child_node.state.board.reshape((1, 1, 8, 8))
        if self.use_policy or self.use_value:
            probs, value = self.model.predict(board)
            value = value[0][0]
            probs = probs.reshape((8, 8))
            if child_node.state.turn == -1:
                probs = probs.T
            child_node.prior_policy = probs
        if not self.use_value:
            value = self._simulate(child_node)
        child_node.value = value
        self.visited_nodes[state] = child_node
        return child_node

    def _select(self, parent_node):
        '''returns node with max UCB Weight'''
        children = parent_node.children
        items = children.items()
        if not self.use_policy:
            UCB_weights = [(v.UCBWeight(parent_node.visits, self.UCB_const, parent_node.state.turn), v) for k,v in items]
        else:
            UCB_weights = [(v.UCBWeight_noPolicy(parent_node.visits, self.UCB_const, parent_node.state.turn), v) for k,v in items]
        # choose the action with max UCB
        node = max(UCB_weights, key=lambda c: c[0])
        return node[1]



    def _simulate(self, next_node):
        # returns outcome of simulated playout
        state = next_node.state
        while not state.isTerminal:
            available_moves = state.availableMoves
            index = choice(range(len(available_moves)))
            move = available_moves[index]
            state = state.makeMove(move)
        return (state.winner + 1) / 2

    def _backprop(self, selected_node, root_node, outcome):
        current_node = selected_node
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
    def __init__(self, model, rollouts=1600, save_tree=True, competitive=False):
        self.name = "AlphaHex"
        self.bestModel = model
        self.rollouts = rollouts
        self.MCTS = None
        self.save_tree = save_tree
        self.competitive = competitive
    def getMove(self, game):
        if self.MCTS is None:
            self.MCTS = MCTS(self.bestModel)
        if game in self.MCTS.visited_nodes:
            root_node = self.MCTS.visited_nodes[game]
        else:
            root_node = self.MCTS.expand(game, 1, None)
        self.MCTS.runSearch(root_node, self.rollouts)
        searchProbabilities = self.MCTS.getSearchProbabilities(root_node)
        moves = list(searchProbabilities.keys())
        probs = list(searchProbabilities.values())
        prob_items = searchProbabilities.items()
        print(probs)
        # if competitive play, choose highest prob move
        if self.competitive:
            best_move = max(prob_items, key=lambda c: c[1])
            return best_move[0]
        # else if self-play, choose stochastically
        else:
            chosen_idx = choice(len(moves), p=probs)
            return moves[chosen_idx]
