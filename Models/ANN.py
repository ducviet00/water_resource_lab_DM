from keras.layers import Conv1D, Input, Bidirectional, LSTM, Concatenate, Reshape, TimeDistributed, Dense
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint

#NOTE: link to the original paper: https://www.tandfonline.com/doi/full/10.1080/02626667.2015.1083650#aHR0cHM6Ly93d3cudGFuZGZvbmxpbmUuY29tL2RvaS9wZGYvMTAuMTA4MC8wMjYyNjY2Ny4yMDE1LjEwODM2NTA/bmVlZEFjY2Vzcz10cnVlQEBAMA==


def model_builder(input_dim=5, output_dim=2, target_timestep=1, dropout=0.1):
    input = Input(shape=(input_dim, ))

    dense_1 = Dense(units=10, activation='relu')
    dense_1_out = dense_1(input)

    dense_3 = Dense(units=output_dim)
    output = dense_3(dense_1_out)

    model = Model(inputs=input, outputs=output)

    model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mape'])
    model.summary()
    return model