from Hex import *
from AlphaHex import  DeepLearningPlayer
from keras.models import load_model
import numpy as np

def formatTrainingData(training_data):
    """ training data is an array of tuples (boards, probs, value), we need to reshape into np array of state boards for x, and list of two np arrays of search probs and value for y"""
    x = []
    y_values = []
    y_probs = []
    for (board, probs, value) in training_data:
        x.append(board)
        y_probs.append(probs)
        y_values.append(value)

    # use subset of training data
    train_x = np.array(x).reshape((len(x), 1, 8, 8))
    train_y = {'policy_out': np.array(y_probs).reshape((len(y_probs), 64)), 'value_out': np.array(y_values)}
    return train_x, train_y

def reshapedSearchProbs(search_probs):
    moves = list(search_probs.keys())
    probs = list(search_probs.values())
    reshaped_probs = np.zeros(64).reshape(8,8)
    for move, prob in zip(moves, probs):
        reshaped_probs[move[0]][move[1]] = prob
    return reshaped_probs.reshape(64)

def trainModel(current_model, training_data, iteration):
    new_model = current_model
    train_x, train_y = formatTrainingData(training_data)
    np.savez('training_data_'+str(iteration), train_x, train_y['policy_out'], train_y['value_out'])
    #TODO: save training data to npz
    new_model.fit(train_x, train_y, verbose = 1, validation_split=0.2, epochs = 10, shuffle=True)
    new_model.save('new_model_iteration_' + str(iteration) + '.h5')
    return new_model

def evaluateModel(new_model, current_model, iteration):
    numEvaluationGames = 40
    newChallengerWins = 0
    threshold = 0.55

    # play 400 games between best and latest models
    for i in range(int(numEvaluationGames//2)):
        g = HexGame(8)  
        game, _ = play_game(g, DeepLearningPlayer(new_model, rollouts=400), DeepLearningPlayer(current_model, rollouts=400), False)
        if game.winner:
            newChallengerWins += game.winner
    for i in range(int(numEvaluationGames//2)):
        g = HexGame(8)
        game, _ = play_game(g, DeepLearningPlayer(current_model, rollouts=400), DeepLearningPlayer(new_model, rollouts=400), False)
        if game.winner == -1:
            newChallengerWins += game.winner
    winRate = newChallengerWins/numEvaluationGames
    print('evaluation winrate' + str(winRate))
    text_file = open("evaluation_results.txt", "w")
    text_file.write("Evaluation results for iteration" + str(iteration) + ": " + str(winRate) + '\n')
    text_file.close()
    if winRate >= threshold:
        new_model.save('current_best_model.h5')

def play_game(game, player1, player2, show=True):
    """Plays a game then returns the final state."""
    new_game_data = []
    while not game.isTerminal:
        if show:
            print(game)
        if game.turn == 1:
            m = player1.getMove(game)
        else:
            m = player2.getMove(game)
        if m not in game.availableMoves:
            raise Exception("invalid move: " + str(m))
        node = player1.MCTS.visited_nodes[game]
        if game.turn == 1:
            search_probs = player1.MCTS.getSearchProbabilities(node)
            board = game.board
        if game.turn == -1:
            search_probs = player2.MCTS.getSearchProbabilities(node)
            board = -game.board.T
        reshaped_search_probs = reshapedSearchProbs(search_probs)    
        if game.turn == -1:
            reshaped_search_probs = reshaped_search_probs.reshape((8,8)).T.reshape(64)

        if np.random.random() > 0.5:
            new_game_data.append((board, reshaped_search_probs, None))
        if np.random.random() > 0.5:
            new_game_data.append((board, reshaped_search_probs, None))
        game = game.makeMove(m)
    if show:
        print(game, "\n")

        if game.winner != 0:
            print("player", print_char(game.winner), "(", end='')
            print((player1.name if game.winner == 1 else player2.name)+") wins")
        else:
            print("it's a draw")
    outcome = 1 if game.winner == 1 else 0
    new_training_data = [(board, searchProbs, outcome) for (board, searchProbs, throwaway) in new_game_data]
    # add training data
    # training_data += new_training_data
    return game, new_training_data

def selfPlay(current_model, numGames, training_data):
    for i in range(numGames):
        print('Game #: ' + str(i))
        g = HexGame(8)
        player1 = DeepLearningPlayer(current_model, rollouts=400)
        player2 = DeepLearningPlayer(current_model, rollouts=400)
        # player2 = DeepLearningPlayer(current_model)
        game, new_training_data = play_game(g, player1, player2, False)
        training_data+= new_training_data
    return training_data

for i in range(10):
    training_data = []
    current_model = load_model('current_best_model.h5')
    training_data = selfPlay(current_model, 100, training_data)
    new_model = trainModel(current_model, training_data, i)
    evaluateModel(new_model, current_model, i)
