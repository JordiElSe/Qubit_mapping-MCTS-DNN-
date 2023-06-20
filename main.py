import logging
from qiskit.providers.fake_provider import FakeCasablancaV2
from Trainer import Trainer
import sys
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, Measure
from qiskit.transpiler import Target, InstructionProperties
from qiskit.circuit.library import CXGate, UGate
import networkx as nx
from qiskit.circuit import ClassicalRegister, QuantumCircuit, CircuitInstruction
from qiskit.circuit import Reset
from qiskit.circuit.library import standard_gates
from qiskit.circuit.exceptions import CircuitError
logger = logging.getLogger(__name__)
#Set the logging level to INFO
logger.setLevel(logging.DEBUG)

# Create a console handler
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)

# Create a formatter
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(ch)


args = {
    'numIters': 1,            #Number of times the circuit is changed
    'numEps': 15,              # Number of games with the same circuit and target
    'maxDepth': 4,
    'initialPositions': 10,    # Number of initial positions to be generated inside an episode
    'maxMoves': 4,           # Maximum number of moves in a game
    'tempThreshold': 0,        # Number of moves to be made before temperature is set to 0 (ie. deterministic play based on visit count)
    'numMCTSSims': 800,          # Number of games moves for MCTS to simulate.
    'numQubits': 5,             # Maximum number of qubits in the circuit
    'Depth': 7,             # Maximum depth of the circuits
    'c1': 5.0,
    'c2': 19.5,
    'dirichletAlpha': 0.35,
    'epsilon': 0.15,
    'checkpoint': './temp/',
    'load_model': False,
    'load_circuit': False,
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 30,
    'batch_size': 16,
    'cuda': False,
    'num_filters': 512,
}

def main():

    logger.info('Loading the Trainer and FakeAthens backend...')
    t = Trainer(args, target=generate_fake_athens_target())
    logger.info('Starting the learning process 🎉')
    model = t.learn()
    model.save_checkpoint(folder=args['checkpoint'], filename='finalModel.h5')

def generate_fake_athens_target():
    '''
    Returns a target with n_qubits that can execute any single qubit and CNOT gate in both directions
    '''
    target = Target(num_qubits=5)
    # Add random edges until the graph is connected
    two_qubit_connections = [(0,1),(1,0),(1,2),(2,1),(2,3),(3,2),(3,4),(4,3)]

    #Add the rotation gates and the C-NOT gate to the target
    target.add_instruction(
        UGate(Parameter('theta'), Parameter('phi'), Parameter('lam')),
        {
            (x,): InstructionProperties(error=.00001, duration=5e-8) for x in range(5)
        }
    )
    target.add_instruction(CXGate(), {c : None for c in two_qubit_connections})
    target.add_instruction(
        Measure(),
        {
            (x,): InstructionProperties(error=.001, duration=5e-5) for x in range(5)
        }
    )
    
    #display(target.build_coupling_map().draw())

    return target
        

if __name__ == "__main__":
    main()