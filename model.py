from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam


def create_fc_model(layers=None):
    if layers is None:
        layers = [4, 2, 2, 1]

    model = Sequential()
    model.add(Dense(layers[1], input_shape=(layers[0],), activation='relu'))
    for layer in layers[2:-1]:
        model.add(Dense(layer, activation='relu'))

    model.add(Dense(layers[-1], activation='sigmoid'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=.1), metrics=['accuracy'])

    return model


def create_generation(n_individuals=10, layers=None):
    generation = []
    for _ in range(n_individuals):
        generation.append(create_fc_model(layers))

    return generation
