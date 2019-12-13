from keras import layers
from keras import models
import numpy as np
from model import create_fc_model


def mutate(model, severity):
    for layer in model.layers:
        weights = layer.get_weights()

        #print(weights)
        #print(np.random.normal(0, 0.05*severity))

        for j, w1 in enumerate(weights[0]):
            for k, w2 in enumerate(weights[0][j]):
                #print("j: ", j, " k: ", k)
                #print(weights[0][j][k])
                weights[0][j][k] += np.random.normal(0, 0.05 * severity)
                #print(weights[0][j][k])

        #print(weights)


model = create_fc_model()
mutate(model, 1)

