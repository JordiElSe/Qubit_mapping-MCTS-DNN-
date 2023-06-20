import logging
from tqdm import tqdm
from MCTS import MCTS
import numpy as np
import os
import h5py
from Game import Game
from NNet import NNet
import sys
""" import random
from qiskit.circuit import Parameter, Measure
from qiskit.transpiler import Target, InstructionProperties
from qiskit.circuit.library import CXGate, UGate
import networkx as nx """



log = logging.getLogger(__name__)
#Set the logging level to INFO
log.setLevel(logging.DEBUG)

# Create a console handler
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)

# Create a formatter
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

# Add the handler to the logger
log.addHandler(ch)

class Trainer:
    def __init__(self, args, target):
        self.args = args
        self.target = target
    
    def executeEpisode(self, game, nn):
        '''
        Executes one game of self-play until the end
        '''
        training_examples = []
        #mapping = game.getSabreMapping()
        mapping = game.getTrivialMapping()
        """ for i in range(self.args['initialPositions']):
            if i == self.args['initialPositions'] - 1:
                mapping = game.getSabreMapping()
            else: """
        #mapping = game.getRandomMapping()
        move = 0
        mcts = MCTS(game, nn, self.args)
        game_states = [mapping]
        actions = []
        #trajectory = [mapping]
        while True:
            move += 1
            log.info(f'Move #{move}')
            temp = int(move < self.args['tempThreshold'])

            pi = mcts.getActionProb(mapping, temp=temp,game_states=game_states)
            z = mcts.getZValue(mapping)

            aug_data = game.augmentData(mapping,pi,z)
            training_examples.extend(aug_data)

            action = np.random.choice(len(pi), p=pi)
            actions.append(action)
            mapping = game.getNextState(mapping, action)
            #trajectory.extend([action, mapping])
            #log.info(f'After move #{move} the reduced depth is: {game.getTerminalValue(mapping)} ')

            #If the maximum depth is reached or we chose the action do nothing -> GAME OVER
            if self.args['maxMoves'] == move or action == game.getActionSize()-1:
                log.debug(f'Game Over. Chosen actions: {actions}. Terminal depth: {game.getDepth(mapping)}')
                game.setCustomMapping(mapping)
                return training_examples
                """ log.debug('According to the MCTS the best solution is:')
                for i in range(0,len(trajectory)-2,2):
                    log.debug(f'From {trajectory[i]} taking action {trajectory[i+1]} -> {trajectory[i+2]} terminal value: {game.getTerminalValue(trajectory[i+2])}')
                log.debug('According to the NN the best solution is:')
                mapping = trajectory[0]
                input_nn = game.getMatrixFromMapping(mapping)
                nn.train(training_examples, game)
                pi, v = nn.predict(input_nn)
                for _ in range (self.args['maxMoves']):
                    action = np.argmax(pi)
                    if action == game.getActionSize()-1:
                        log.debug(f'From {mapping} chosen action is do nothing with probability {pi[action]} -> Game Over with terminal value: {game.getTerminalValue(mapping)}')
                        break
                    log.debug(f'From {mapping} with a predicted value {v} taking action {action} with probability {pi[action]} -> {game.getNextState(mapping, action)} terminal value: {game.getTerminalValue(game.getNextState(mapping, action))}')
                    mapping = game.getNextState(mapping, action)
                    input_nn = game.getMatrixFromMapping(mapping)
                    pi, v = nn.predict(input_nn)
                break """
        #return training_examples       

    def learn(self):
        '''
        Executes numIters iterations with numEps episodes of self-play
        in each iteration. The neural network is update continually as
        episodes are executed rather than waiting for an itertion to complete
        '''
        if self.args['load_model']:
            log.info('Loading Neural Network...')
            nn = NNet(self.args)
            nn.load_checkpoint_weights(folder=self.args['checkpoint'], filename='model.h5')
        else:
            log.info('Initializing Neural Network...')
            nn = NNet(self.args)
        #target = self.generate_random_non_directional_target(self.args['numQubits'])
        for i in range(1, self.args['numIters'] + 1):
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')

            log.info('Initializing Game...')
            g = Game(self.args, self.target)

            # execute episodes
            # save the current model every 5 episodes
            for j in tqdm(range(self.args['numEps']), desc='Self Play'):
                log.info(f'Starting Episode #{j+1} ...')
                training_examples = self.executeEpisode(g, nn)
                log.info('Updating Neural Network...')
                nn.train(training_examples, g)
                self.storeTrainingExamples(training_examples, episode=j+1, iteration=i)
                nn.save_checkpoint(folder=self.args['checkpoint'])
                """ for _ in range (2):
                    sabre_mapping = g.getSabreMapping()
                    sabre_input = g.getMatrixFromMapping(sabre_mapping)
                    sabre_pi, sabre_v = nn.predict(sabre_input)
                    print(f'With SABRE the depth is: {g.original_depth} and the values is {sabre_v}')
                    for _ in range (self.args['maxDepth']):
                        action = np.argmax(sabre_pi)
                        print(f'Selecting action {action} with probability {sabre_pi[action]}')
                        if action == g.getActionSize()-1:
                            break
                        sabre_mapping = g.getNextState(sabre_mapping, action)
                        sabre_input = g.getMatrixFromMapping(sabre_mapping)
                        sabre_pi, sabre_v = nn.predict(sabre_input)
                    print(f'With the NN and starting from SABRE the depth is: {g.getTerminalValue(sabre_mapping)} and the values is {sabre_v}')

                    
                    trivial_mapping = g.getTrivialMapping()
                    trivial_input = g.getMatrixFromMapping(trivial_mapping)
                    trivial_pi, trivial_v = nn.predict(trivial_input)
                    print(f'With the trivial mapping the depth is: {g.original_depth} and the values is {trivial_v}')
                    for _ in range (self.args['maxDepth']):
                        action = np.argmax(trivial_pi)
                        print(f'Selecting action {action} with probability {trivial_pi[action]}')
                        if action == g.getActionSize()-1:
                            break
                        trivial_mapping = g.getNextState(trivial_mapping, action)
                        trivial_input = g.getMatrixFromMapping(trivial_mapping)
                        trivial_pi, trivial_v = nn.predict(trivial_input)
                    print(f'With the NN and starting from the trivial mapping the depth is: {g.getTerminalValue(trivial_mapping)} and the values is {trivial_v}')
                    
                    random_mapping = g.getRandomMapping()
                    random_input = g.getMatrixFromMapping(random_mapping)
                    random_pi, random_v = nn.predict(random_input)
                    print(f'With a random mapping the depth is: {g.original_depth} and the values is {random_v}')    
                    for _ in range (self.args['maxDepth']):
                        action = np.argmax(random_pi)
                        print(f'Selecting action {action} with probability {random_pi[action]}')
                        if action == g.getActionSize()-1:
                            break
                        random_mapping = g.getNextState(random_mapping, action)
                        random_input = g.getMatrixFromMapping(random_mapping)
                        random_pi, random_v = nn.predict(random_input)
                    print(f'With the NN and starting from a random mapping the depth is: {g.getTerminalValue(random_mapping)} and the values is {random_v}')
                input("Press Enter to continue...") """
        return nn
    
    def storeTrainingExamples(self, training_examples, episode, iteration):
        '''
        Stores the training examples in a HDF5 file format in the folder specified in args['checkpoint']
        '''
        folder = self.args['checkpoint']
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, f'trainExamples_Ep_{episode}_It_{iteration}.hdf5')
        with h5py.File(filename, 'w') as f:
            f.create_dataset('mapping', data = [x[0] for x in training_examples])
            f.create_dataset('pi', data = [x[1] for x in training_examples])
            f.create_dataset('z', data = [x[2] for x in training_examples])

    def loadTrainingExamples(self, episode):
        folder = self.args['checkpoint']
        filename = os.path.join(folder, f'trainExamples_Ep_{episode}.hdf5')
        with h5py.File(filename, 'r') as f:
            mapping = f['mapping'][:]
            pi = f['pi'][:]
            z = f['z'][:]
        return zip(mapping, pi, z)

    """ @staticmethod
    def generate_random_non_directional_target(n_qubits):
        '''
        Returns a target with n_qubits that can execute any single qubit and CNOT gate in both directions
        '''
        target = Target(num_qubits=n_qubits)

        #List of all possible edges between qubits
        two_qubit_connections = []
        for i in range(n_qubits):
            for j in range(i+1, n_qubits):
                two_qubit_connections.append((i, j))
        

        # Create an empty graph with n nodes
        G = nx.Graph()
        G.add_nodes_from(range(n_qubits))

        # Add random edges until the graph is connected
        edges = []
        while not nx.is_connected(G):
            # Choose a random connection and remove it from the list
            u, v = random.choice(two_qubit_connections)
            two_qubit_connections.remove((u, v))
            edges.append((u, v))
            edges.append((v, u))
            # Add the edge between the two nodes
            G.add_edge(u, v)

        #Get connections of the target
        two_qubit_connections= edges

        #Add the rotation gates and the C-NOT gate to the target
        target.add_instruction(
            UGate(Parameter('theta'), Parameter('phi'), Parameter('lam')),
            {
                (x,): InstructionProperties(error=.00001, duration=5e-8) for x in range(n_qubits)
            }
        )
        target.add_instruction(CXGate(), {c : None for c in two_qubit_connections})
        target.add_instruction(
            Measure(),
            {
                (x,): InstructionProperties(error=.001, duration=5e-5) for x in range(n_qubits)
            }
        )
        
        #display(target.build_coupling_map().draw())

        return target """
    
        