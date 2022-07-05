import tensorflow as tf
import numpy as np

from keras.layers import Layer

class WeightedAverage(Layer):
    def __init__(self, weights, **kargs):
        super(WeightedAverage, self).__init__(**kargs)
        if type(weights) is not list or len(weights) <= 1:
            raise Exception("WeightAverage() :  must be initialized on a list of weight tensors. Got : ", weights)

        self.inputShape = np.array(weights[0]).shape       
        self.nInputs = len(weights)

        self.w = []
        for weight in weights:
            self.w.append(tf.convert_to_tensor(weight))
            if self.inputShape != self.w[-1].shape:
                raise Exception("WeightedAverage() : The shapes of weights and biases(if given) must be identical.")
    
    def call(self, inputs):
        if type(inputs) is not list or len(inputs) != self.nInputs:
            raise Exception("WeightAverage must be called on a list of input tensors with same size of nInputs. Got : " + inputs)
        self.weightedInputs = []
        for i in range(self.nInputs):
            if inputs[i].shape[1:] != self.inputShape:
                raise Exception("call : Size of input tensor must be equal to that of weight.", inputs[i].shape, self.inputShape)
            self.weightedInputs.append(tf.multiply(self.w[i], inputs[i]))
        return tf.math.divide(tf.math.accumulate_n(self.weightedInputs), tf.math.accumulate_n(self.w))


if __name__ == "__main__":
    x1, x2 = tf.ones((2, 2), dtype="float32"), tf.zeros((2, 2), dtype="float32")
    av = WeightedAverage([[[2., 2.], [2., 2.]], [[1., 1.], [1., 1.]]])
    y = av([x1, x2])
    print(y)