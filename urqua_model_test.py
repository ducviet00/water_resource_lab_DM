import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def data_processing(link):
    dat = pd.read_csv(link, sep='\t', header=None)
    discharge = dat.iloc[:, 2] / 35.315  #convert from ft3 to m3
    discharge = discharge.dropna(axis=0)
    discharge = discharge.to_numpy()
    # plt.plot(discharge)
    # plt.show()

    # print(discharge.tail())
    return discharge.reshape(-1, 1)


#def build_and_train(en_x_train, de_x_train, de_y_train):
def build_and_train(x_train, y_train, mode):
    from Models.multi_rnn_cnn import model_builder, train_model

    model = model_builder(input_dim=1, output_dim=1, target_timestep=1)
    if mode == 'train':
        model, _ = train_model(model,
                               x_train=x_train,
                               y_train=y_train,
                               batch_size=128,
                               epochs=300,
                               patience=50,
                               early_stop=True,
                               save_dir='./')
    else:
        model.load_weights('./best_model_300.hdf5')
    # from keras.models import Model
    # from keras.layers import Input, Conv1D, Bidirectional, LSTM, Dense, TimeDistributed, Concatenate
    # encoder_inputs = Input(shape=(None, 1))
    # # conv1d = Conv1D(filters=16,kernel_size=2,strides=1,padding='same')
    # # conv_out = conv1d(encoder_inputs)
    # encoder = Bidirectional(
    #     LSTM(128, return_state=True, return_sequences=False))
    # encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder(
    #     encoder_inputs)

    # state_h = Concatenate()([forward_h, backward_h])
    # state_c = Concatenate()([forward_c, backward_c])
    # encoder_states = [state_h, state_c]

    # # decoder
    # decoder_inputs = Input(shape=(None, 1))
    # #de_in_concat = Concatenate(axis=-1)([decoder_inputs,encoder_outputs])
    # decoder_lstm_1 = LSTM(256,
    #                       return_sequences=True,
    #                       return_state=False,
    #                       dropout=0)
    # decoder_outputs_1 = decoder_lstm_1(decoder_inputs,
    #                                    initial_state=encoder_states)
    # #dc_input = Concatenate(axis=-1)([decoder_inputs,decoder_outputs_1])

    # #decoder_dense_1 = Dense(units=64,activation='relu')(dc_input)

    # decoder_dense = TimeDistributed(Dense(units=1))
    # decoder_outputs = decoder_dense(decoder_outputs_1)

    # model = Model(inputs=[encoder_inputs, decoder_inputs],
    #               outputs=decoder_outputs)

    # #optimizer = SGD(lr=1e-6, momentum=0.9,decay=self.lr_decay,nesterov=True)
    # #optimizer = RMSprop(learning_rate=5e-3)
    # #optimizer = Adadelta(rho=0.95)
    # #optimizer = Adam(learning_rate=5e-2,amsgrad=False)
    # model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mape'])

    # from keras.callbacks import EarlyStopping, ModelCheckpoint
    # callbacks = []
    # #lr_schedule = LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch / 20))
    # early_stop = EarlyStopping(monitor='val_loss',
    #                            patience=50,
    #                            restore_best_weights=True)
    # checkpoint = ModelCheckpoint('./' + 'best_model.hdf5',
    #                              monitor='val_loss',
    #                              verbose=1,
    #                              save_best_only=True)

    # #callbacks.append(lr_schedule)
    # callbacks.append(early_stop)
    # callbacks.append(checkpoint)

    # history = model.fit(x=[en_x_train, de_x_train],
    #                     y=de_y_train,
    #                     batch_size=128,
    #                     epochs=300,
    #                     callbacks=callbacks,
    #                     validation_split=0.2)

    return model


# def predict_and_eval(model, scaler, en_x_test, de_x_test, de_y_test, shape):
#     y_pre = model.predict([en_x_test, de_x_test])
def predict_and_eval(model, scaler, x_test, y_test, shape):
    y_pre = model.predict(x_test)
    #retransform
    mask = np.zeros(shape=shape)
    test_shape = y_test.shape[0]

    mask[-test_shape:, 0] = y_test[:, 0, 0]
    actual_data = scaler.inverse_transform(mask)[-test_shape:, 0]

    mask[-test_shape:, 0] = y_pre[:, 0, 0]
    actual_predict = scaler.inverse_transform(mask)[-test_shape:, 0]

    plt.plot(actual_data, label='ground_truth')
    plt.plot(actual_predict, label='predict')
    plt.legend(loc='best')
    plt.show()

    #evaluate
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    variance_score_q = r2_score(actual_data, actual_predict)
    mse_q = mean_squared_error(actual_data, actual_predict)
    mae_q = mean_absolute_error(actual_data, actual_predict)

    print(f'Model: R2: {variance_score_q} MSE: {mse_q} MAE: {mae_q} \n')


if __name__ == '__main__':
    dat = pd.read_csv('./ProcessedData/urqua_daily.csv', header=0,
                      index_col=0).iloc[:, 2]
    dat = dat.to_numpy().reshape(-1, 1)
    # from ProcessingData.reprocess_daily import ed_extract_data
    # en_x, de_x, de_y, scaler = ed_extract_data(dat,
    #                                            cols_x=[0],
    #                                            cols_y=[0],
    #                                            mode='min_max')

    # split_index = 1000
    # en_x_train, en_x_test = en_x[:-split_index], en_x[-split_index:]
    # de_x_train, de_x_test = de_x[:-split_index], de_x[-split_index:]
    # de_y_train, de_y_test = de_y[:-split_index], de_y[-split_index:]

    # model = build_and_train(en_x_train, de_x_train, de_y_train)
    # predict_and_eval(model, scaler, en_x_test, de_x_test, de_y_test, dat.shape)

    from ProcessingData.reprocess_daily import extract_data
    x, y, scaler = extract_data(dat, cols_x=[0], cols_y=[0], mode='min_max')

    split_index = 1000
    x_train, y_train = x[:-split_index], y[:-split_index]
    x_test, y_test = x[-split_index:], y[-split_index:]

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default='train', type=str, help='Run mode.')
    args = parser.parse_args()

    model = build_and_train(x_train, y_train, args.mode)
    predict_and_eval(model, scaler, x_test, y_test, dat.shape)
    #NOTE: R2: 0.8308290131125085 MSE: 8269.315980151623 MAE: 30.480127354232394 - encoder decoder
    #  R2: 0.8223683988075663 MSE: 8682.882717338583 MAE: 29.756075936874076 - multi-rnn-cnn
