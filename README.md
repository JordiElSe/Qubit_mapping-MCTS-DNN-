# Qubit Mapping with a DNN aided with MCTS

This repository contains and implementation of the AlphaZero-like algorithm for Qubit Mapping. As describre in (https://arxiv.org/abs/1712.01815), the algorithm trains a neural network to predict the policy and value of a given state, and it uses an MCTS to plan the next move. Episodes are executed to generate training data, and the neural network is trained with this data.

The code contains funcitons to generate random circuits with CNOT and U gates and a specific number of qubits, and to generate random processors with a specified number of qubits that can only execute CNOT and U gates. 

To execute the code, run the following command:

```
python3 main.py
```

The code is set to train for quantum circuits of 5 qubits and depth 7 and the 5 qubit processor FakeAthensV2 from qiskit. This parameter and hyperparameters can be changed in the main.py file. If you want to train with randomly generated processors, use the function generate_random_non_directional_target() in the Trainer.py file.

This code is inspired by:
https://github.com/suragnair/alpha-zero-general/tree/master
