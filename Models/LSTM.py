from keras.layers import Conv1D, Input, Bidirectional, LSTM, Concatenate, Reshape, TimeDistributed, Dense
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint

#NOTE: link to the original paper: https://iopscience.iop.org/article/10.1088/1755-1315/299/1/012037/pdf


def model_builder(input_dim=5, output_dim=2, target_timestep=1, dropout=0.1):
    input = Input(shape=(None, input_dim))

    rnn_3 = LSTM(units=256, return_sequences=False, return_state=False, dropout=dropout, recurrent_dropout=dropout)
    rnn_out_3 = rnn_3(input)

    reshape_l = Reshape(target_shape=(target_timestep, -1))
    rnn_out = reshape_l(rnn_out_3)

    dense_3 = TimeDistributed(Dense(units=output_dim))
    output = dense_3(rnn_out)

    model = Model(inputs=input, outputs=output)

    model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mape'])
    model.summary()
    return model