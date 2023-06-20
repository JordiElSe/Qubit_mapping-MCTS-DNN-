import numpy as np
from NNet import NNet
from Game import Game
from qiskit.providers.fake_provider import FakeCasablancaV2

args = {
    'numIters': 25,
    'numEps': 10,              # Number of complete self-play games to simulate during a new iteration.
    'maxDepth': 6,        
    'tempThreshold': 4,        # Number of moves to be made before temperature is set to 0 (ie. deterministic play based on visit count)
    'numMCTSSims': 400,          # Number of games moves for MCTS to simulate.
    'numQubits': 7,             # Maximum number of qubits in the circuit
    'Depth': 8,             # Maximum depth of the circuits
    'c1': 4.0,
    'c2': 19.5,
    'dirichletAlpha': [0.3, 0.15, 0.03],
    'epsilon': 0.25,
    'checkpoint': './temp/',
    'load_model': False,
    'lr': 0.002,
    'dropout': 0.3,
    'epochs': 30,
    'batch_size': 16,
    'cuda': False,
    'num_filters': 512,
}

#load the model from ./temp/model.h5
model = NNet(args)
model.load_checkpoint_model(folder='./temp/', filename='model.h5')
#model.load_checkpoint(folder='./temp/', filename='model.h5')
for i in range (2):
    print(f'================= At iteration {i} these are the results =====================')
    game = Game(args, target=FakeCasablancaV2().target)
    
    sabre_mapping = game.getSabreMapping()
    sabre_input = game.getMatrixFromMapping(sabre_mapping)
    sabre_pi, sabre_v = model.predict(sabre_input)
    print(f'With SABRE the depth is: {game.original_depth} and the values is {sabre_v}')
    for _ in range (args['maxDepth']):
        action = np.argmax(sabre_pi)
        if action == game.getActionSize()-1:
            break
        sabre_mapping = game.getNextState(sabre_mapping, action)
        sabre_input = game.getMatrixFromMapping(sabre_mapping)
        sabre_pi, sabre_v = model.predict(sabre_input)
    print(f'With the NN and starting from SABRE the depth is: {game.getTerminalValue(sabre_mapping)} and the values is {sabre_v}')

    
    trivial_mapping = game.getTrivialMapping()
    trivial_input = game.getMatrixFromMapping(trivial_mapping)
    trivial_pi, trivial_v = model.predict(trivial_input)
    print(f'With the trivial mapping the depth is: {game.original_depth} and the values is {trivial_v}')
    for _ in range (args['maxDepth']):
        action = np.argmax(trivial_pi)
        if action == game.getActionSize()-1:
            break
        trivial_mapping = game.getNextState(trivial_mapping, action)
        trivial_input = game.getMatrixFromMapping(trivial_mapping)
        trivial_pi, trivial_v = model.predict(trivial_input)
    print(f'With the NN and starting from the trivial mapping the depth is: {game.getTerminalValue(trivial_mapping)} and the values is {trivial_v}')
    
    #random_mapping = game.getRandomMapping()
    random_mapping = (4, 0, 6, 2, 5, 3, 1)
    random_input = game.getMatrixFromMapping(random_mapping)
    random_pi, random_v = model.predict(random_input)
    print(f'With a random mapping the depth is: {game.original_depth} and the values is {random_v}')    
    for _ in range (args['maxDepth']):
        action = np.argmax(random_pi)
        if action == game.getActionSize()-1:
            break
        random_mapping = game.getNextState(random_mapping, action)
        random_input = game.getMatrixFromMapping(random_mapping)
        random_pi, random_v = model.predict(random_input)
    print(f'With the NN and starting from a random mapping the depth is: {game.getTerminalValue(random_mapping)} and the values is {random_v}')
    
    