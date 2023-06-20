from qiskit import transpile, QuantumCircuit
import numpy as np
import random
""" from qiskit.circuit import Parameter, Measure
from qiskit.transpiler import Target, InstructionProperties
from qiskit.circuit.library import CXGate, UGate
import networkx as nx """
from qiskit.circuit import ClassicalRegister, QuantumCircuit, CircuitInstruction, Reset
from qiskit.circuit.library import standard_gates
from qiskit.circuit.exceptions import CircuitError

class Game:
    def __init__(self, args, target):
        self.args = args
        if args['load_circuit']:
            with open('my_circuit.qasm', 'r') as file:
                loaded_qasm_str = file.read()
            self.qc = QuantumCircuit.from_qasm_str(loaded_qasm_str)
        else:
            self.qc = Game.random_circuit(num_qubits=args['numQubits'], depth=args['Depth']-1, measure=True) #depth without the measurement
            with open('my_circuit.qasm', 'w') as file:
                file.write(self.qc.qasm())
        #self.target = Game.generate_random_non_directional_target(n_qubits=args['numQubits'])
        self.target = target
        self.actions = [(i,j) for i in range(args['numQubits']) for j in range(i+1,args['numQubits'])]
        self.actions.append((0,0))
        self.qcMatrix = self.getMatrixFromCircuit()
        self.original_depth = None

    def getSabreMapping(self):
        best_translation = transpile(self.qc, target=self.target, layout_method='sabre', routing_method='sabre', optimization_level=3)
        for _ in range(15):
            translated_circuit = transpile(self.qc, target=self.target, layout_method='sabre', routing_method='sabre', optimization_level=3)
            if translated_circuit.depth() < best_translation.depth():
                best_translation = translated_circuit
        self.original_depth = best_translation.depth()
        print(f'Depth with sabre mapping: {self.original_depth}')
        mapping_list = [0] * self.args['numQubits']
        for key, value in best_translation._layout.initial_layout._v2p.items():
            #generate list of length equal the number of qubits in the circuit and position key contains the value
            #Only if qubit is not an ancilla
            try:
                mapping_list[best_translation.find_bit(key).index] = value
            except:
                pass
        return tuple(mapping_list)
    
    def getTrivialMapping(self):
        best_translation = transpile(self.qc, target=self.target, layout_method='trivial', routing_method='sabre', optimization_level=3)
        for _ in range(15):
            translated_circuit = transpile(self.qc, target=self.target, layout_method='trivial', routing_method='sabre', optimization_level=3)
            if translated_circuit.depth() < best_translation.depth():
                best_translation = translated_circuit
        self.original_depth = best_translation.depth()
        print(f'Depth with trivial mapping: {self.original_depth}')
        return tuple(range(self.args['numQubits']))
    
    def getRandomMapping(self):
        #Choose random permutation of range list from 0 to numQubits
        mapping = list(np.random.permutation(self.args["numQubits"]))
        best_translation = transpile(self.qc, target=self.target, initial_layout=mapping, routing_method='sabre', optimization_level=3)
        for _ in range(15):
            translated_circuit = transpile(self.qc, target=self.target, initial_layout=mapping, routing_method='sabre', optimization_level=3)
            if translated_circuit.depth() < best_translation.depth():
                best_translation = translated_circuit
        self.original_depth = best_translation.depth()
        print(f'Depth with random mapping: {self.original_depth}')
        return tuple(mapping)
    
    def setCustomMapping(self, mapping):
        self.original_depth = float('inf')
        for _ in range(15):
             self.original_depth = min(self.original_depth, transpile(self.qc, target=self.target, initial_layout=list(mapping), routing_method='sabre', optimization_level=3).depth())

    def getDepth(self, mapping):
        return min([transpile(self.qc, target=self.target, initial_layout=list(mapping), routing_method='sabre', optimization_level=3).depth() for _ in range(15)])

    def getMatrixSize(self):
        return self.args['numQubits'], self.args['Depth']
    
    def getNextState(self, s, a):
        s = list(s)
        s[self.actions[a][0]], s[self.actions[a][1]] = s[self.actions[a][1]], s[self.actions[a][0]]
        return tuple(s)
    
    def getActionSize(self):
        return len(self.actions)

    def getTerminalValue(self, s):
        '''
        Transpile the quantum circuit 10 times using the current state as the initial layout and get the minimum depth
        '''
        depths = []
        for _ in range(10):
            depths.append(transpile(self.qc, target=self.target, initial_layout=list(s), optimization_level=3, routing_method='sabre').depth())
        if self.original_depth <= min(depths):
            return -1.0
        return float(self.original_depth - min(depths))

    def getMatrixFromMapping(self, mapping):
        new_matrix = np.zeros(self.qcMatrix.shape)
        for i in range(self.qcMatrix.shape[0]):
            new_matrix[i] = self.qcMatrix[mapping[i]]
        return new_matrix

    def getMatrixFromCircuit(self):
        
        '''
        Converts a quantum circuit composed of only U gates (rotations), CNOT gates and measurements into a matrix of ints.
        The matrix is of size (n_qubits, depth) and each element i,j contains the gate executed by qubit i at depth j.
        If no gate is executed by qubit i at depth j, the element is 0.
        If a CNOT gate is executed by qubits i and j at depth k, both the element i,k and j,k contain the integer 85
        If a U gate is executed by qubit i at depth j, the element i,j contains the integer 170.
        If a measurement is executed by qubit i at depth j, the element i,j contains the integer 255 
        '''

        #Initialize matrix of ints to -1
        matrix = np.full((self.args['numQubits'],self.args['Depth']),0.0)
        #Iterate over all the gates in the circuit
        for gate in self.qc.data:
            if gate[0].name == 'measure':
                #Get qubit index
                qubit = self.qc.find_bit(gate[1][0]).index
                #Get index of column where qubit is -1
                d = np.where(matrix[qubit,:] == 0)[0][0]
                #Set matrix element
                matrix[qubit,d] = 1.0
            elif gate[0].name == 'cx':
                #Get qubit indices
                qubit1 = self.qc.find_bit(gate[1][0]).index
                qubit2 = self.qc.find_bit(gate[1][1]).index
                #Get the first column where both qubit1 and qubit2 are -1
                d = np.where((matrix[qubit1,:] == 0) & (matrix[qubit2,:] == 0))[0][0]
                #Set matrix elements
                matrix[qubit1,d] = 85*1.0/255
                matrix[qubit2,d] = 85*1.0/255
            elif gate[0].name == 'u':
                #Get qubit index
                qubit = self.qc.find_bit(gate[1][0]).index
                #Get index of first column where qubit is -1
                d = np.where(matrix[qubit,:] == 0)[0][0]
                #Set matrix element
                matrix[qubit,d] = 170*1.0/255
            elif gate[0].name == 'barrier':
                pass
            else:
                raise ValueError(f'Gate {gate[0].name} not supported')
            matrix = np.where(matrix == 1.0, 1.0, 0.0)
        return matrix

    def augmentData(self, mapping, pi, z):
        matrix = self.getMatrixFromMapping(mapping)
        data = [(matrix, pi, z)]
        """ for _ in range(100):        
            new_matrix = np.copy(matrix)
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    if matrix[i][j] != 85*1.0/255:
                        #Choose a random new value 
                        new_value = random.choice([0.0, 170*1.0/255, 1.0])
                        #Set the gate in the matrix
                        new_matrix[i][j] = new_value
            data.append((new_matrix, pi, z)) """
        return data

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
    
    @staticmethod
    def random_circuit(num_qubits, depth, max_operands=4, measure=False, conditional=False, reset=False, seed=None):
        """Generate random circuit of arbitrary size and form.
        This function will generate a random circuit by randomly selecting gates
        from the set of standard gates in :mod:`qiskit.extensions`. For example:
        .. plot::
        :include-source:
        from qiskit.circuit.random import random_circuit
        circ = random_circuit(2, 2, measure=True)
        circ.draw(output='mpl')
        Args:
            num_qubits (int): number of quantum wires
            depth (int): layers of operations (i.e. critical path length)
            max_operands (int): maximum qubit operands of each gate (between 1 and 4)
            measure (bool): if True, measure all qubits at the end
            conditional (bool): if True, insert middle measurements and conditionals
            reset (bool): if True, insert middle resets
            seed (int): sets random seed (optional)
        Returns:
            QuantumCircuit: constructed circuit
        Raises:
            CircuitError: when invalid options given
        """
        if num_qubits == 0:
            return QuantumCircuit()
        if max_operands < 1 or max_operands > 4:
            raise CircuitError("max_operands must be between 1 and 4")
        max_operands = max_operands if num_qubits > max_operands else num_qubits

        gates_1q = [ (standard_gates.UGate, 1, 3) ]
        if reset:
            gates_1q.append((Reset, 1, 0))
        gates_2q = [(standard_gates.CXGate, 2, 0)]

        gates = gates_1q.copy()
        if max_operands >= 2:
            gates.extend(gates_2q)
        gates = np.array(
            gates, dtype=[("class", object), ("num_qubits", np.int64), ("num_params", np.int64)]
        )
        gates_1q = np.array(gates_1q, dtype=gates.dtype)

        qc = QuantumCircuit(num_qubits)

        if measure or conditional:
            cr = ClassicalRegister(num_qubits, "c")
            qc.add_register(cr)

        if seed is None:
            seed = np.random.randint(0, np.iinfo(np.int32).max)
        rng = np.random.default_rng(seed)

        qubits = np.array(qc.qubits, dtype=object, copy=True)

        # Apply arbitrary random operations in layers across all qubits.
        for _ in range(depth):
            # We generate all the randomness for the layer in one go, to avoid many separate calls to
            # the randomisation routines, which can be fairly slow.

            # This reliably draws too much randomness, but it's less expensive than looping over more
            # calls to the rng. After, trim it down by finding the point when we've used all the qubits.
            gate_specs = rng.choice(gates, size=len(qubits))
            cumulative_qubits = np.cumsum(gate_specs["num_qubits"], dtype=np.int64)
            # Efficiently find the point in the list where the total gates would use as many as
            # possible of, but not more than, the number of qubits in the layer.  If there's slack, fill
            # it with 1q gates.
            max_index = np.searchsorted(cumulative_qubits, num_qubits, side="right")
            gate_specs = gate_specs[:max_index]
            slack = num_qubits - cumulative_qubits[max_index - 1]
            if slack:
                gate_specs = np.hstack((gate_specs, rng.choice(gates_1q, size=slack)))

            # For efficiency in the Python loop, this uses Numpy vectorisation to pre-calculate the
            # indices into the lists of qubits and parameters for every gate, and then suitably
            # randomises those lists.
            q_indices = np.empty(len(gate_specs) + 1, dtype=np.int64)
            p_indices = np.empty(len(gate_specs) + 1, dtype=np.int64)
            q_indices[0] = p_indices[0] = 0
            np.cumsum(gate_specs["num_qubits"], out=q_indices[1:])
            np.cumsum(gate_specs["num_params"], out=p_indices[1:])
            parameters = rng.uniform(0, 2 * np.pi, size=p_indices[-1])
            rng.shuffle(qubits)

            # We've now generated everything we're going to need.  Now just to add everything.  The
            # conditional check is outside the two loops to make the more common case of no conditionals
            # faster, since in Python we don't have a compiler to do this for us.
            if conditional:
                is_conditional = rng.random(size=len(gate_specs)) < 0.1
                condition_values = rng.integers(
                    0, 1 << min(num_qubits, 63), size=np.count_nonzero(is_conditional)
                )
                c_ptr = 0
                for gate, q_start, q_end, p_start, p_end, is_cond in zip(
                    gate_specs["class"],
                    q_indices[:-1],
                    q_indices[1:],
                    p_indices[:-1],
                    p_indices[1:],
                    is_conditional,
                ):
                    operation = gate(*parameters[p_start:p_end])
                    if is_cond:
                        # The condition values are required to be bigints, not Numpy's fixed-width type.
                        operation.condition = (cr, int(condition_values[c_ptr]))
                        c_ptr += 1
                    qc._append(CircuitInstruction(operation=operation, qubits=qubits[q_start:q_end]))
            else:
                for gate, q_start, q_end, p_start, p_end in zip(
                    gate_specs["class"], q_indices[:-1], q_indices[1:], p_indices[:-1], p_indices[1:]
                ):
                    operation = gate(*parameters[p_start:p_end])
                    qc._append(CircuitInstruction(operation=operation, qubits=qubits[q_start:q_end]))

        if measure:
            qc.measure(qc.qubits, cr)

        return qc