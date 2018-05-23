import numpy as np
from hex_zero_model import build_model

def load_data(filename):
    hex_data = np.load(filename)

    states = hex_data['states']
    turns = hex_data['turns']
    visits = hex_data['visits']
    moves = hex_data['moves']
    values = hex_data['values']

    for i in range(states.shape[0]):
        if turns[i] == -1:
            states[i] = states[i].T
            moves[i] = np.array([[moves[i][1], moves[i][0]]])
            visits[i] = visits[i].T
            values[i] = 1 - values[i].T

    # reshape data for model (channels first)
    states = states.reshape(states.shape[0], 1, 8, 8)

    # train_X = states[:4*states.shape[0] // 5]
    # test_X = states[4*states.shape[0] // 5:]
    train_X = states

    probabilities = calculate_probabilities(visits)
    y_values = calculate_values(moves, values)

    training_probs = probabilities[:4*probabilities.shape[0] // 5]
    training_values = y_values[:4*y_values.shape[0] // 5]
    testing_probs = probabilities[4*y_values.shape[0] // 5:]
    testing_values = y_values[4*y_values.shape[0] // 5:]
    
    train_Y = {'policy_out':probabilities, 'value_out':y_values}
    # test_Y = {'policy_out':testing_probs, 'value_out':testing_values}

    return train_X, train_Y

def calculate_probabilities(visits):
    normalize_sums = visits.sum(axis=1).sum(axis=1)
    reshaped = visits.reshape((visits.shape[0], visits.shape[1]*visits.shape[2]))

    normalized = reshaped/normalize_sums[:,None]

    probabilities = normalized.reshape((visits.shape[0], visits.shape[1]*visits.shape[2]))

    return probabilities

def calculate_values(moves, values):
    y_values = np.array([value[move[0]][move[1]] for move, value in zip(moves, values)])
    return y_values


train_X, train_Y = load_data('hex_data.npz')
model = build_model()
history = model.fit(train_X, train_Y, verbose = 1, validation_split=0.2, epochs = 25, shuffle=True)

# loss, accuracy = model.evaluate(test_X, test_Y, verbose = 1)
# print("accuracy: {}%".format(accuracy*100))

model.save('new_supervised_zero.h5')
