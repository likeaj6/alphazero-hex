## Read More Here:
https://notes.jasonljin.com/projects/2018/05/20/Training-AlphaZero-To-Play-Hex.html:

## Training the neural network
`hex_zero_model.py` contains the building of the Deep Neural Network used for policy and value prediction.
`sl_bootstrap.py` contains a script to bootstrap the neural network on existing hex data, calling on hex_zero_model to build the neural net before training the neural net for the specified epochs.

### Instructions
`python3 sl_bootstrap.py`

## Evaluating against various players
`Hex.py` contains several functions to play against different players (Self, Random, HexPlayerBryce), where you can specify the number of games and who's player 1, and whether to show the game turn by turn. 

### Instructions
`python3 Hex.py`

## AlphaHex Agent
`AlphaHex.py` contains the actual agent that utilizes the general AlphaZero algorithm. 

## Self-Play & Reinforcement Learning
`TrainAlphaHexZero.py` contains a script to self-play a specified number of iterations. In each iteration, the AlphaHex agent plays a specified number of games against itself, where it collects randomly 50% of the game data played in the iteration and saves it into a .npz file. It then trains the current best model on this game data for a specified number of epochs, and evaluates the new model against this previous model for a specified number of iterations, where the results are written to a .txt file. If the win rate is over a set threshold, than the new model will become the new current best model to be used in the next iteration of self play.  
### Instructions
`python3 TrainAlphaHexZero.py`
