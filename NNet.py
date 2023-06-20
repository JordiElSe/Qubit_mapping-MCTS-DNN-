from modelArchitecture import modelArchitecture
import numpy as np
import os
import functools
from keras.models import load_model
import tensorflow as tf

class NNet:
    def __init__(self, args):
        self.args = args
        self.nnet = modelArchitecture(args)
        
    def train(self, examples, game):
        '''
        examples: list of examples, each example is of form (mapping, pi, z)
        '''
        input_circuit, target_pis, target_z = list(zip(*examples))
        #input circuit is a tuple of arrays
        #create a new array that contains each array on the tuple as rows
        input_circuit = np.stack(input_circuit, axis=0)
        target_pis = np.asarray(target_pis)
        target_z = np.asarray(target_z)
        self.nnet.model.fit(x = input_circuit, y = [target_pis, target_z], batch_size = self.args['batch_size'], epochs = self.args['epochs'], verbose=2, shuffle=True)

    def predict(self, circuit):
        '''
        circuit: np array with the circuit encoded
        '''
        # run
        circuit = circuit[np.newaxis, :, :]
        pi, v = self.nnet.model.predict(circuit, batch_size=1, verbose=1)
        return pi[0], v[0][0]
    
    def save_checkpoint(self, folder='checkpoint', filename='model.h5'):
        '''
        Save the current model in the folder specified in args as 'filename'
        '''
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        self.nnet.model.save(filepath)
        filename = 'weights_' + filename
        filepath = os.path.join(folder, filename)
        self.nnet.model.save_weights(filepath)

    def load_checkpoint_model(self, folder='checkpoint', filename='model.h5'):
        '''
        Load the current model from the folder specified in args as 'filename'
        '''
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("No model in path {}".format(filepath))
        topk_acc = functools.partial(tf.keras.metrics.top_k_categorical_accuracy, k=self.args['numQubits']-1)

        topk_acc.__name__ = 'topk_acc'
        self.nnet.model = load_model(filepath, custom_objects={'topk_acc': topk_acc})


    def load_checkpoint_weights(self, folder='checkpoint', filename='model.h5'):
        '''
        Load the current model from the folder specified in args as 'filename'
        '''
        filename = 'weights_' + filename
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("No model in path {}".format(filepath))
        self.nnet.model.load_weights(filepath)