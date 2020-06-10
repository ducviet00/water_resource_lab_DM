from keras.layers import Conv1D, Input, Bidirectional, LSTM, Concatenate, Reshape, TimeDistributed, Dense
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint


def model_builder(input_dim=7, output_dim=2, target_timestep=1, dropout=0.1):
    en_x = Input(shape=(None, input_dim))
    de_x = Input(shape=(None, output_dim))

    # conv = Conv1D(filters=16, kernel_size=2, strides=1, padding='same')
    # conv_out = conv(input)
    # conv_2 = Conv1D(filters=32, kernel_size=3, padding='same')
    # conv_out_2 = conv_2(conv_out)
    # conv_3 = Conv1D(filters=64, kernel_size=4, padding='same')
    # conv_out_3 = conv_3(conv_out_2)

    encoder = Bidirectional(
        LSTM(units=128,
             return_sequences=False,
             return_state=True,
             dropout=dropout,
             recurrent_dropout=dropout,
             name='encoder'))
    en_out, forward_h, forward_c, backward_h, backward_c = encoder(en_x)
    state_h = Concatenate(axis=-1)([forward_h, backward_h])
    state_c = Concatenate(axis=-1)([forward_c, backward_c])

    decoder = LSTM(units=256,
                   return_sequences=False,
                   return_state=False,
                   dropout=dropout,
                   recurrent_dropout=dropout,
                   name='decoder')
    de_out = decoder(de_x, initial_state=[state_h, state_c])

    reshape_l = Reshape(target_shape=(target_timestep, -1))
    rnn_out = reshape_l(de_out)

    dense_3 = TimeDistributed(Dense(units=output_dim))
    output = dense_3(rnn_out)

    model = Model(inputs=[en_x, de_x], outputs=output)

    model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mape'])

    from keras.utils.vis_utils import plot_model
    plot_model(model, to_file='./Log/Ensemble/sub_model.png')

    return model


def train_model(model,
                en_x_train,
                de_x_train,
                y_train,
                batch_size,
                epochs,
                fraction=0.1,
                patience=0,
                early_stop=False,
                save_dir=''):
    callbacks = []

    checkpoint = ModelCheckpoint(save_dir + f'ed_best_model_{epochs}.hdf5',
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True)
    callbacks.append(checkpoint)

    #early_stop = epochs == 250
    if (early_stop):
        early_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        callbacks.append(early_stop)

    history = model.fit(x=[en_x_train, de_x_train],
                        y=y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_split=fraction)

    return model, history