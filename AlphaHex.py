from math import log, sqrt
from numpy.random import choice
from numpy import array
import numpy as np
import sys

class Node(object):
    """Node used in MCTS"""
    def __init__(self, state, parent_node, prior_prob):
        self.state = state
        self.children = {} # maps moves to Nodes
        self.visits = 0
        self.value = 0.5
        # self.value = 0.5 if parent_node is None else parent_node.value
        self.prior_prob = prior_prob
        self.prior_policy = np.zeros((8, 8))
        self.parent_node = parent_node

    def updateValue(self, outcome, debug=False):
        """Updates the value estimate for the node's state."""
        if debug:
            print('visits: ', self.visits)
            print('before value: ', self.value)
            print('outcome: ', outcome)
        self.value = (self.visits*self.value + outcome)/(self.visits+1)
        self.visits += 1
        if debug:
            print('updated value:', self.value)
    def UCBWeight_noPolicy(self, parent_visits, UCB_const, player):
        if player == -1:
            return (1-self.value) + UCB_const*sqrt(parent_visits)/(1+self.visits)
        else:
            return self.value + UCB_const*sqrt(parent_visits)/(1+self.visits)
    def UCBWeight(self, parent_visits, UCB_const, player):
        """Weight from the UCB formula used by parent to select a child."""
        if player == -1:
            return (1-self.value) + UCB_const*self.prior_prob/(1+self.visits)
        else:
            return self.value + UCB_const*self.prior_prob/(1+self.visits)

class MCTS:
    def __init__(self, model, UCB_const=2, use_policy=True, use_value=True):
        self.visited_nodes = {} # maps state to node
        self.model = model
        self.UCB_const = UCB_const
        self.use_policy = use_policy
        self.use_value = use_value

    def runSearch(self, root_node, num_searches):
        # start search from root
        for i in range(num_searches):
            selected_node = root_node
            available_moves = selected_node.state.availableMoves
            # if we've already explored this node, continue down path until we reach a node we haven't expanded yet by selecting node w/ largest UCB weight
                # select node that maximizes Upper Confidence Bound
            while len(available_moves) == len(selected_node.children) and not selected_node.state.isTerminal:
                if selected_node == root_node:
                    selected_node = self._select(selected_node, debug=False)
                else:
                    selected_node = self._select(selected_node, debug=False)
                available_moves = selected_node.state.availableMoves
            if not selected_node.state.isTerminal:
                if self.use_policy:
                    if selected_node.state not in self.visited_nodes:
                        selected_node = self.expand(selected_node, debug=False)
                    outcome = selected_node.value
                    if root_node.state.turn == -1:
                        outcome = 1-outcome
                    self._backprop(selected_node, root_node, outcome, debug=False)
                else:
                    moves = selected_node.state.availableMoves
                    np.random.shuffle(moves)
                    for move in moves:
                        if not selected_node.state.makeMove(move) in self.nodes:
                            break
            else:
                outcome = 1 if selected_node.state.winner == 1 else 0
                self._backprop(selected_node, root_node, outcome)
    def create_children(self, parent_node):
        if len(parent_node.state.availableMoves) != len(parent_node.children):
            for move in parent_node.state.availableMoves:
                next_state = parent_node.state.makeMove(move)
                child_node = Node(next_state, parent_node, parent_node.prior_policy[move[0]][move[1]])
                # print(parent_node.prior_policy[move[0]][move[1]])
                parent_node.children[move] = child_node
    def _select(self, parent_node, debug=False):
        '''returns node with max UCB Weight'''
        # print(parent_node.prior_policy)
        # if len(parent_node.state.availableMoves) != len(parent_node.children):
        #    for move in parent_node.state.availableMoves:
        #        next_state = parent_node.state.makeMove(move)
        #        child_node = Node(next_state, parent_node, parent_node.prior_policy[move[0]][move[1]])
        #        # print(parent_node.prior_policy[move[0]][move[1]])
        #        parent_node.children[move] = child_node
        children = parent_node.children
        items = children.items()
        if self.use_policy:
            UCB_weights = [(v.UCBWeight(parent_node.visits, self.UCB_const, parent_node.state.turn), v) for k,v in items]
        else:
            UCB_weights = [(v.UCBWeight_noPolicy(parent_node.visits, self.UCB_const, parent_node.state.turn), v) for k,v in items]
        # if debug:
        #   print([k for k, v in UCB_weights])
        # sys.exit(1)
        # choose the action with max UCB
        node = max(UCB_weights, key=lambda c: c[0])
        if debug:
            print('weight:', node[0])
            print('move:', node[1].state)
            print('value:', node[1].value)
            print('visits:', node[1].visits)
        return node[1]

    def modelPredict(self, state):
        if state.turn == -1:
            board = (-state.board).T.reshape((1, 1, 8, 8))
        else:
            board = state.board.reshape((1, 1, 8, 8))
        probs, value = self.model.predict(board)
        value = value[0][0]
        probs = probs.reshape((8, 8))
        if state.turn == -1:
            probs = probs.T
        return probs, value

    def expandRoot(self, state):
        root_node = Node(state, None, 1)
        if self.use_policy or self.use_value:
            probs, value = self.modelPredict(state)
            root_node.prior_policy = probs
        if not self.use_value:
            value = self._simulate(root_node)
        root_node.value = value
        self.visited_nodes[state] = root_node
        self.create_children(root_node)
        return root_node

    def expand(self, selected_node, debug=False):
        # policy = [selected_node.prior_policy[move] for move in selected_node.state.availableMoves]
        # move = selected_node.state.availableMoves[policy.index(max(policy))]
        # next_state = selected_node.state.makeMove(move)
        # child_node = Node(next_state, selected_node, selected_node.prior_policy[move])
        if self.use_policy or self.use_value:
            probs, value = self.modelPredict(selected_node.state)
            selected_node.prior_policy = probs
        if not self.use_value:
            # select randomly
            value = self._simulate(selected_node)
        if debug:
            print('expanding node', selected_node.state)
        selected_node.value = value
        self.visited_nodes[selected_node.state] = selected_node
        self.create_children(selected_node)
        return selected_node





    def _simulate(self, next_node):
        # returns outcome of simulated playout
        state = next_node.state
        while not state.isTerminal:
            available_moves = state.availableMoves
            index = choice(range(len(available_moves)))
            move = available_moves[index]
            state = state.makeMove(move)
        return (state.winner + 1) / 2

    def _backprop(self, selected_node, root_node, outcome, debug=False):
        current_node = selected_node
        # print(outcome)
        if selected_node.state.isTerminal:
            outcome = 1 if selected_node.state.winner == 1 else 0
        while current_node != root_node:
            if debug:
                print('selected_node: ', selected_node.state)
                print('outcome: ', outcome)
                print('backpropping')
            current_node.updateValue(outcome, debug=False)
            current_node = current_node.parent_node
            # print(current_node.visits)
        # update root node
        root_node.updateValue(outcome)

    def getSearchProbabilities(self, root_node):
        children = root_node.children
        items = children.items()
        child_visits = [child.visits for action, child in items]
        sum_visits = sum(child_visits)
        # print(child_visits)
        if sum_visits != 0:
            normalized_probs = {action: (child.visits/sum_visits) for action, child in items}
        else:
            normalized_probs = {action: (child.visits/len(child_visits)) for action, child in items}
        return normalized_probs

class DeepLearningPlayer:
    def __init__(self, model, rollouts=1600, save_tree=True, competitive=False):
        self.name = "AlphaHex"
        self.bestModel = model
        # self.player = player
        self.rollouts = rollouts
        self.MCTS = None
        self.save_tree = save_tree
        self.competitive = competitive
    def getMove(self, game):
        if self.MCTS is None or not self.save_tree:
            self.MCTS = MCTS(self.bestModel)
        if self.save_tree and game in self.MCTS.visited_nodes:
            root_node = self.MCTS.visited_nodes[game]
        else:
            root_node = self.MCTS.expandRoot(game)
        self.MCTS.runSearch(root_node, self.rollouts)
        searchProbabilities = self.MCTS.getSearchProbabilities(root_node)
        moves = list(searchProbabilities.keys())
        probs = list(searchProbabilities.values())
        prob_items = searchProbabilities.items()
        print(probs)
        # if competitive play, choose highest prob move
        if self.competitive:
            best_move = max(prob_items, key=lambda c: c[1])
            print(best_move)
            # sys.exit(1)
            return best_move[0]
        # else if self-play, choose stochastically
        else:
            chosen_idx = choice(len(moves), p=probs)
            return moves[chosen_idx]
