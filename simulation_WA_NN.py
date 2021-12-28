from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
# np.random.seed(1337)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import regularizers

#****************************************custom layer*************************************************
class Threshold(layers.Layer):
    def __init__(self, units = 1):
        super(Threshold, self).__init__()
        self.unit = units
        self.w_initializer = tf.keras.initializers.Constant(value=50)

    def build(self, input_shape):
        self.w = self.add_weight(name='w',shape=(input_shape[-1],self.unit),
                                 # trainable=True,
                                 # initializer='random_normal',
                                 trainable=False,
                                 initializer=self.w_initializer
                                 )
        self.alpha = self.add_weight(name='alpha',shape=(input_shape[-1],self.unit),
                                     initializer='random_normal',
                                     # initializer='he_normal',
                                     trainable=True)

    def call(self, inputs):

        return (inputs - tf.transpose(self.alpha)) * tf.transpose(self.w)


class Simulation_50(tf.keras.Model):
    def __init__(self):
        super(Simulation_50, self).__init__()
        self.rule1_0 = Threshold()
        self.rule2_0 = Threshold()
        self.rule3_0 = Threshold()
        self.rule4_0 = Threshold()
        self.rule5_0 = Threshold()

        #, kernel_regularizer=regularizers.l1(0.001)

        self.rule1_2 = layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1(0.001))
        self.rule2_2 = layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1(0.001))
        self.rule3_2 = layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1(0.001))
        self.rule4_2 = layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1(0.001))
        self.rule5_2 = layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1(0.001))
        # , use_bias=False

        self.outlayer = layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1(0.001))
        # ,,kernel_regularizer = regularizers.l2(0.001)

    def call(self, inputs, delta=0):
        r1_0 = self.rule1_0(tf.gather(inputs,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],axis=1))
        r1_0 = tf.nn.sigmoid(r1_0)

        r2_0 = self.rule2_0(tf.gather(inputs, [10, 11, 12, 13, 14, 15, 16, 17, 18, 19], axis=1))
        r2_0 = tf.nn.sigmoid(r2_0)

        r3_0 = self.rule3_0(tf.gather(inputs, [20, 21, 22, 23, 24, 25, 26, 27, 28, 29], axis=1))
        r3_0 = tf.nn.sigmoid(r3_0)

        r4_0 = self.rule4_0(tf.gather(inputs, [30, 31, 32, 33, 34, 35, 36, 37, 38, 39], axis=1))
        r4_0 = tf.nn.sigmoid(r4_0)

        r5_0 = self.rule5_0(tf.gather(inputs, [40, 41, 42, 43, 44, 45, 46, 47, 48, 49], axis=1))
        r5_0 = tf.nn.sigmoid(r5_0)

        z1_1 = self.rule1_2(r1_0)
        z2_1 = self.rule2_2(r2_0)
        z3_1 = self.rule3_2(r3_0)
        z4_1 = self.rule4_2(r4_0)
        z5_1 = self.rule5_2(r5_0)

        x = layers.concatenate([z1_1, z2_1, z3_1, z4_1, z5_1])
        x = self.outlayer(x)

        return x


class Simulation_8(tf.keras.Model):
    def __init__(self):
        super(Simulation_8, self).__init__()
        self.rule1_0 = Threshold()
        self.rule2_0 = Threshold()
        self.rule3_0 = Threshold()
        self.rule4_0 = Threshold()

        #, kernel_regularizer=regularizers.l1(0.001)

        self.rule1_2 = layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1(0.001))
        self.rule2_2 = layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1(0.001))
        self.rule3_2 = layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1(0.001))
        self.rule4_2 = layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1(0.001))
        # , use_bias=False, kernel_regularizer=regularizers.l1(0.001)

        self.outlayer = layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1(0.001))
        # ,,kernel_regularizer = regularizers.l2(0.001)

    def call(self, inputs, delta=0):
        r1_0 = self.rule1_0(tf.gather(inputs,[0, 1],axis=1))
        r1_0 = tf.nn.sigmoid(r1_0)

        r2_0 = self.rule2_0(tf.gather(inputs, [2, 3], axis=1))
        r2_0 = tf.nn.sigmoid(r2_0)

        r3_0 = self.rule3_0(tf.gather(inputs, [4, 5], axis=1))
        r3_0 = tf.nn.sigmoid(r3_0)

        r4_0 = self.rule4_0(tf.gather(inputs, [6, 7], axis=1))
        r4_0 = tf.nn.sigmoid(r4_0)

        z1_1 = self.rule1_2(r1_0)
        z2_1 = self.rule2_2(r2_0)
        z3_1 = self.rule3_2(r3_0)
        z4_1 = self.rule4_2(r4_0)

        x = layers.concatenate([z1_1, z2_1, z3_1, z4_1])
        x = self.outlayer(x)

        return x


class Simulation_32(tf.keras.Model):
    def __init__(self):
        super(Simulation_32, self).__init__()
        self.rule1_0 = Threshold()
        self.rule2_0 = Threshold()
        self.rule3_0 = Threshold()
        self.rule4_0 = Threshold()

        #, kernel_regularizer=regularizers.l1(0.001)

        self.rule1_2 = layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1(0.001))
        self.rule2_2 = layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1(0.001))
        self.rule3_2 = layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1(0.001))
        self.rule4_2 = layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1(0.001))
        # , use_bias=False, kernel_regularizer=regularizers.l1(0.001)

        self.outlayer = layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1(0.001))
        # ,,kernel_regularizer = regularizers.l2(0.001)

    def call(self, inputs, delta=0):
        r1_0 = self.rule1_0(tf.gather(inputs,[0, 1, 2, 3, 4, 5, 6, 7],axis=1))
        r1_0 = tf.nn.sigmoid(r1_0)

        r2_0 = self.rule2_0(tf.gather(inputs, [8, 9, 10, 11, 12, 13, 14, 15], axis=1))
        r2_0 = tf.nn.sigmoid(r2_0)

        r3_0 = self.rule3_0(tf.gather(inputs, [16, 17, 18, 19, 20, 21, 22, 23], axis=1))
        r3_0 = tf.nn.sigmoid(r3_0)

        r4_0 = self.rule4_0(tf.gather(inputs, [24, 25, 26, 27, 28, 29, 30, 31], axis=1))
        r4_0 = tf.nn.sigmoid(r4_0)

        z1_1 = self.rule1_2(r1_0)
        z2_1 = self.rule2_2(r2_0)
        z3_1 = self.rule3_2(r3_0)
        z4_1 = self.rule4_2(r4_0)

        x = layers.concatenate([z1_1, z2_1, z3_1, z4_1])
        x = self.outlayer(x)

        return x


class Simulation_64(tf.keras.Model):
    def __init__(self):
        super(Simulation_64_2, self).__init__()
        self.rule1_0 = Threshold()
        self.rule2_0 = Threshold()
        self.rule3_0 = Threshold()
        self.rule4_0 = Threshold()
        self.rule5_0 = Threshold()
        self.rule6_0 = Threshold()
        self.rule7_0 = Threshold()
        self.rule8_0 = Threshold()

        #, kernel_regularizer=regularizers.l1(0.001)
        self.rule1_1 = layers.Dense(2, activation='selu', kernel_regularizer=regularizers.l2(0.001))
        self.rule2_1 = layers.Dense(2, activation='selu', kernel_regularizer=regularizers.l2(0.001))
        self.rule3_1 = layers.Dense(2, activation='selu', kernel_regularizer=regularizers.l2(0.001))
        self.rule4_1 = layers.Dense(2, activation='selu', kernel_regularizer=regularizers.l2(0.001))
        self.rule5_1 = layers.Dense(2, activation='selu', kernel_regularizer=regularizers.l2(0.001))
        self.rule6_1 = layers.Dense(2, activation='selu', kernel_regularizer=regularizers.l2(0.001))
        self.rule7_1 = layers.Dense(2, activation='selu', kernel_regularizer=regularizers.l2(0.001))
        self.rule8_1 = layers.Dense(2, activation='selu', kernel_regularizer=regularizers.l2(0.001))

        self.rule1_2 = layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001))
        self.rule2_2 = layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001))
        self.rule3_2 = layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001))
        self.rule4_2 = layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001))
        self.rule5_2 = layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001))
        self.rule6_2 = layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001))
        self.rule7_2 = layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001))
        self.rule8_2 = layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001))
        # , use_bias=False, kernel_regularizer=regularizers.l1(0.001)

        self.outlayer = layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1(0.001))
        # ,,kernel_regularizer = regularizers.l2(0.001)

    def call(self, inputs, delta=0):
        r1_0 = self.rule1_0(tf.gather(inputs,[0, 1, 2, 3, 4, 5, 6, 7],axis=1))
        r1_0 = tf.nn.sigmoid(r1_0)

        r2_0 = self.rule2_0(tf.gather(inputs, [8, 9, 10, 11, 12, 13, 14, 15], axis=1))
        r2_0 = tf.nn.sigmoid(r2_0)

        r3_0 = self.rule3_0(tf.gather(inputs, [16, 17, 18, 19, 20, 21, 22, 23], axis=1))
        r3_0 = tf.nn.sigmoid(r3_0)

        r4_0 = self.rule4_0(tf.gather(inputs, [24, 25, 26, 27, 28, 29, 30, 31], axis=1))
        r4_0 = tf.nn.sigmoid(r4_0)

        r5_0 = self.rule5_0(tf.gather(inputs,[32, 33, 34, 35, 36, 37, 38, 39],axis=1))
        r5_0 = tf.nn.sigmoid(r5_0)

        r6_0 = self.rule6_0(tf.gather(inputs, [40, 41, 42, 43, 44, 45, 46, 47], axis=1))
        r6_0 = tf.nn.sigmoid(r6_0)

        r7_0 = self.rule7_0(tf.gather(inputs, [48, 49, 50, 51, 52, 53, 54, 55], axis=1))
        r7_0 = tf.nn.sigmoid(r7_0)

        r8_0 = self.rule8_0(tf.gather(inputs, [56, 57, 58, 59, 60, 61, 62, 63], axis=1))
        r8_0 = tf.nn.sigmoid(r8_0)

        z1_1 = self.rule1_1(r1_0)
        z2_1 = self.rule2_1(r2_0)
        z3_1 = self.rule3_1(r3_0)
        z4_1 = self.rule4_1(r4_0)
        z5_1 = self.rule5_1(r5_0)
        z6_1 = self.rule6_1(r6_0)
        z7_1 = self.rule7_1(r7_0)
        z8_1 = self.rule8_1(r8_0)

        z1_2 = self.rule1_2(z1_1)
        z2_2 = self.rule2_2(z2_1)
        z3_2 = self.rule3_2(z3_1)
        z4_2 = self.rule4_2(z4_1)
        z5_2 = self.rule5_2(z5_1)
        z6_2 = self.rule6_2(z6_1)
        z7_2 = self.rule7_2(z7_1)
        z8_2 = self.rule8_2(z8_1)

        x = layers.concatenate([z1_2, z2_2, z3_2, z4_2, z5_2, z6_2, z7_2, z8_2])
        x = self.outlayer(x)

        return x


class Pfcc_8(tf.keras.Model):
    def __init__(self):
        super(Pfcc_8, self).__init__()
        self.rule1_0 = layers.Dense(1, activation='sigmoid')
        self.rule2_0 = layers.Dense(1, activation='sigmoid')
        self.rule3_0 = layers.Dense(1, activation='sigmoid')
        self.rule4_0 = layers.Dense(1, activation='sigmoid')

        #, kernel_regularizer=regularizers.l1(0.001)

        self.rule1_2 = layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1(0.001))
        self.rule2_2 = layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1(0.001))
        self.rule3_2 = layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1(0.001))
        self.rule4_2 = layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1(0.001))
        # , use_bias=False, kernel_regularizer=regularizers.l1(0.001)

        self.outlayer = layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1(0.001))
        # ,,kernel_regularizer = regularizers.l2(0.001)

    def call(self, inputs, delta=0):
        r1_0 = self.rule1_0(tf.gather(inputs,[0, 1],axis=1))
        r2_0 = self.rule2_0(tf.gather(inputs, [2, 3], axis=1))
        r3_0 = self.rule3_0(tf.gather(inputs, [4, 5], axis=1))
        r4_0 = self.rule4_0(tf.gather(inputs, [6, 7], axis=1))

        z1_1 = self.rule1_2(r1_0)
        z2_1 = self.rule2_2(r2_0)
        z3_1 = self.rule3_2(r3_0)
        z4_1 = self.rule4_2(r4_0)

        x = layers.concatenate([z1_1, z2_1, z3_1, z4_1])
        x = self.outlayer(x)
        return x


class Pfcc_32(tf.keras.Model):
    def __init__(self):
        super(Pfcc_32, self).__init__()
        self.rule1_0 = layers.Dense(1, activation='sigmoid')
        self.rule2_0 = layers.Dense(1, activation='sigmoid')
        self.rule3_0 = layers.Dense(1, activation='sigmoid')
        self.rule4_0 = layers.Dense(1, activation='sigmoid')

        self.rule1_2 = layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1(0.001))
        self.rule2_2 = layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1(0.001))
        self.rule3_2 = layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1(0.001))
        self.rule4_2 = layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1(0.001))
        # , use_bias=False, kernel_regularizer=regularizers.l1(0.001)

        self.outlayer = layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1(0.001))
        # ,,kernel_regularizer = regularizers.l2(0.001)

    def call(self, inputs, delta=0):
        r1_0 = self.rule1_0(tf.gather(inputs,[0, 1, 2, 3, 4, 5, 6, 7],axis=1))
        r2_0 = self.rule2_0(tf.gather(inputs, [8, 9, 10, 11, 12, 13, 14, 15], axis=1))
        r3_0 = self.rule3_0(tf.gather(inputs, [16, 17, 18, 19, 20, 21, 22, 23], axis=1))
        r4_0 = self.rule4_0(tf.gather(inputs, [24, 25, 26, 27, 28, 29, 30, 31], axis=1))

        z1_1 = self.rule1_2(r1_0)
        z2_1 = self.rule2_2(r2_0)
        z3_1 = self.rule3_2(r3_0)
        z4_1 = self.rule4_2(r4_0)

        x = layers.concatenate([z1_1, z2_1, z3_1, z4_1])
        x = self.outlayer(x)

        return x


class Pfcc_64(tf.keras.Model):
    def __init__(self):
        super(Pfcc_64, self).__init__()
        self.rule1_0 = layers.Dense(1, activation='sigmoid')
        self.rule2_0 = layers.Dense(1, activation='sigmoid')
        self.rule3_0 = layers.Dense(1, activation='sigmoid')
        self.rule4_0 = layers.Dense(1, activation='sigmoid')
        self.rule5_0 = layers.Dense(1, activation='sigmoid')
        self.rule6_0 = layers.Dense(1, activation='sigmoid')
        self.rule7_0 = layers.Dense(1, activation='sigmoid')
        self.rule8_0 = layers.Dense(1, activation='sigmoid')

        #, kernel_regularizer=regularizers.l1(0.001)

        self.rule1_2 = layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001))
        self.rule2_2 = layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001))
        self.rule3_2 = layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001))
        self.rule4_2 = layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001))
        self.rule5_2 = layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001))
        self.rule6_2 = layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001))
        self.rule7_2 = layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001))
        self.rule8_2 = layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001))
        # , use_bias=False, kernel_regularizer=regularizers.l1(0.001)

        self.outlayer = layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1(0.001))
        # ,,kernel_regularizer = regularizers.l2(0.001)

    def call(self, inputs, delta=0):
        r1_0 = self.rule1_0(tf.gather(inputs,[0, 1, 2, 3, 4, 5, 6, 7],axis=1))
        r2_0 = self.rule2_0(tf.gather(inputs, [8, 9, 10, 11, 12, 13, 14, 15], axis=1))
        r3_0 = self.rule3_0(tf.gather(inputs, [16, 17, 18, 19, 20, 21, 22, 23], axis=1))
        r4_0 = self.rule4_0(tf.gather(inputs, [24, 25, 26, 27, 28, 29, 30, 31], axis=1))
        r5_0 = self.rule5_0(tf.gather(inputs,[32, 33, 34, 35, 36, 37, 38, 39],axis=1))
        r6_0 = self.rule6_0(tf.gather(inputs, [40, 41, 42, 43, 44, 45, 46, 47], axis=1))
        r7_0 = self.rule7_0(tf.gather(inputs, [48, 49, 50, 51, 52, 53, 54, 55], axis=1))
        r8_0 = self.rule8_0(tf.gather(inputs, [56, 57, 58, 59, 60, 61, 62, 63], axis=1))

        z1_1 = self.rule1_2(r1_0)
        z2_1 = self.rule2_2(r2_0)
        z3_1 = self.rule3_2(r3_0)
        z4_1 = self.rule4_2(r4_0)
        z5_1 = self.rule5_2(r5_0)
        z6_1 = self.rule6_2(r6_0)
        z7_1 = self.rule7_2(r7_0)
        z8_1 = self.rule8_2(r8_0)

        x = layers.concatenate([z1_1, z2_1, z3_1, z4_1, z5_1, z6_1, z7_1, z8_1])
        x = self.outlayer(x)

        return x