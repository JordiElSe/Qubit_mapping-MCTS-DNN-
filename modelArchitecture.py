from keras.models import Model
from keras.layers import Input, Flatten, Dense
from keras.optimizers import Adam
import tensorflow as tf
import functools

class modelArchitecture:
    def __init__(self, args):
        self.matrix_x = args['numQubits']
        self.matrix_y = args['Depth']
        self.action_size = args['numQubits']*(args['numQubits']-1)/2 + 1
        topk_acc = functools.partial(tf.keras.metrics.top_k_categorical_accuracy, k=self.matrix_x-1)

        topk_acc.__name__ = 'topk_acc'

        def custom_activation(x):
            # Scale the output to the range [0, 1] using the sigmoid function
            scaled_output = tf.keras.activations.sigmoid(x)
            
            # Shift and scale the output to the desired range [-10, -8]
            return -4 + 2 * scaled_output

        input_circuits = Input(shape=(self.matrix_x, self.matrix_y))
        x = Flatten(input_shape=(self.matrix_x, self.matrix_y))(input_circuits)
        x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(4096, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(2048, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        pi = Dense(self.action_size, activation='softmax', name='pi') (x) # batch_size x self.action_size
        v = Dense(1, name='v') (x) # batch_size x 1
        self.model = Model(inputs=input_circuits, outputs=[pi, v])
        self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=Adam(0.0001), metrics=[topk_acc])
