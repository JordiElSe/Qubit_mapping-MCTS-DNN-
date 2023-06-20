import math
import numpy as np
import logging
import sys
import time

EPS = 1e-8

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

class MCTS:

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {} # stores Q values for s,a (as defined in the paper)
        self.Nsa = {} # stores #times edge s,a was visited
        self.Ns = {} # stores #times board s was visited
        self.Ps = {} # stores initial policy (returned by neural net)
        self.game_states = [] # stores the mappings that have been visited in the game so far

    def getActionProb(self, s, temp=1, game_states=[]):
        self.game_states = game_states
        start_time = time.time()
        for i in range(self.args['numMCTSSims']):
            visited_states = []
            log.info(f"MCTS Simulation: {i}")
            
            self.search(s,0, visited_states)
            
            
            #log.debug(f'After simulation {i}:\n Qsa values are:\n {self.Qsa}\n Nsa values are:\n {self.Nsa}\n Ns values are:\n {self.Ns}')
            #Pause executino until user presses continue

        end_time= time.time()
        log.info(f"Time taken for simulation {i}: {end_time-start_time}")
        input("Press Enter to continue...")
            
        
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]
        
        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs
        
        counts = [x**(1./temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        log.debug(f'Probabilities: {probs}')
        return probs
    
    def getZValue(self, s):
        z = 0.0
        counter = 0
        for a in range(self.game.getActionSize()):
            if (s,a) in self.Qsa:
                z += self.Qsa[(s,a)]
                counter += 1
        try:
            return z/counter
        except:
            raise Exception("No actions have been taken from this state")

    def search(self, s, cur_depth=10, visited_states=[]):
        #Check if we are expanding the tree
        log.info(f'Search from state {s} at depth {cur_depth}')
        visited_states.append(s)
        if s not in self.Ps:
            # Leaf node
            circuit = self.game.getMatrixFromMapping(s)
            self.Ps[s], v = self.nnet.predict(circuit)
            self.Ns[s] = 0
            log.debug(f'Expanding new node {s} with value {v}. Initialising backup phase')
            return v.item()
        
        if self.args['maxDepth'] == cur_depth:
            circuit = self.game.getMatrixFromMapping(s)
            #self.Ps[s], _ = self.nnet.predict(circuit)
            v = self.game.getTerminalValue(s)
            self.Ns[s] += 1
            log.debug(f'Maximum depth reached at node {s}. Terminal value {v}.')
            return v


        cur_best = -float('inf')
        best_s = s
        best_a = -1

        for a in range(self.game.getActionSize()):
            next_s = self.game.getNextState(s,a)
            if a != self.game.getActionSize()-1 and (next_s in visited_states or next_s in self.game_states):
                continue
            #If we are at the root we add Dirichlet noise to the prior probabilities
            if cur_depth == 1:
                noise = np.random.dirichlet([self.args['dirichletAlpha']] * self.game.getActionSize())
                p = (1 - self.args['epsilon']) * self.Ps[s][a] + self.args['epsilon'] * noise[a]
            else:
                p = self.Ps[s][a]
            if (s, a) in self.Qsa:
                #Since we are not in a game with rewards inside [0,1] we need to normalise the Qsa values
                uncertainty = (self.args['c1'] + math.log((self.Ns[s] + self.args['c2'] + 1)/self.args['c2']))*p * math.sqrt(self.Ns[s])/(1 + self.Nsa[(s, a)])
                u = self.Qsa[(s, a)] + uncertainty
                log.debug(f'Action {a} from state {s} has value {u} with uncertainty {uncertainty} and Qsa {self.Qsa[(s, a)]}. Visited {self.Nsa[(s, a)]} times')
            else:
                u = (self.args['c1'] + math.log((self.Ns[s] + self.args['c2'] + 1)/self.args['c2']))*p * math.sqrt(self.Ns[s] + EPS)
                log.debug(f'Action {a} from state {s} has value {u}')
            if u > cur_best:
                cur_best = u
                best_s = next_s
                best_a = a
        
        a = best_a
        log.debug(f'Best action is {a} from state {s}')
        #If the chosen action is do nothing then the value Qsa at (s,a) is the terminal value
        if a == self.game.getActionSize()-1:
            #If the action had been taken before then simply increment the counter
            if (s, a) in self.Qsa:
                self.Nsa[(s, a)] += 1
                log.debug(f'Chosen action was do nothing and had already been taken before. The terminal value should be {self.Qsa[(s, a)]}. Initialising backup phase')
            #Otherwise set the value to the terminal value and set the counter to 1
            else:
                self.Qsa[(s, a)] = self.game.getTerminalValue(s)
                self.Nsa[(s, a)] = 1
                log.debug(f'Chosen action was do nothing and had not been taken before. The terminal value is {self.Qsa[(s, a)]}. Initialising backup phase')
            self.Ns[s] += 1
            return self.Qsa[(s, a)]
            
        value = self.search(best_s, cur_depth + 1, visited_states)
        log.debug(f'After searching from state {best_s} the value is {value}. Updating the weights')
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + value) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = value
            self.Nsa[(s, a)] = 1
            log.debug(f'New value for Qsa[{s},{a}] is {self.Qsa[(s, a)]}')

        self.Ns[s] += 1
        return value
