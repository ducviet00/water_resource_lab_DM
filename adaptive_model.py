from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance
import numpy as np
import pandas as pd


def preprocess_discharge(data):
    data.index = pd.to_datetime(data.index)
    filter = (data.index.year >= 1985) & (data.index.year <= 2015)
    data = data[filter]
    # data = data.drop('Dak_to', axis=1)
    data = data.drop('Trung_Nghia', axis=1)

    print(data.head())
    data.to_csv('./RawData/discharge_daily.csv')


def nearest_loc():
    dat = pd.read_csv('./RawData/long-lat.csv', header=0, index_col=0)
    columns_titles = ["lat", "long"]
    dat = dat.reindex(columns=columns_titles)

    import geopy.distance as d
    dic = {}
    kontum = dat.iloc[6]
    for i, r in dat.iterrows():
        dis = d.distance(kontum, r).km
        dic[i] = dis

    print(dic)


#{'Ban _Don': 161.52392479395286,
# 'Cau_14': 191.69793056029073,
# 'Dak_Mot': 44.53592392363741,
# 'Duc _Xuyen': 225.59602064866596,
# 'Giang_Son': 203.174408024674,
# 'Kong_Plong': 20.69787119594489,
# 'Kon_Tum': 0.0,
# 'Krong _buk': 178.75348827184013}
#NOTE:the nearest stations are Kong_Plong, Dak_Mot.


def k_most_related(data):

    kontum = data.iloc[:, 6].to_numpy()
    print(kontum.shape)
    dic = {}

    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean

    for column in data:
        tmp = data[column].to_numpy()
        dic[column], _ = fastdtw(kontum, tmp, dist=euclidean)
        print(dic[column])

    print(dic)


#{'Ban_Don': 1142891.3999999901,
#'Cau_14': 1000087.1000000042,
# 'Dak_Mot': 278022.2020500004,
# 'Duc_Xuyen': 429374.8999999966,
# 'Giang_Son': 349033.19999999995,
# 'Kon_Plong': 384460.4090099976,
# 'Kon_Tum': 0.0,
# 'Krong_Buk': 764518.9000000143}
#NOTE:the most related stations are Kong_Plong, Giang_Son and Dak_Mot.


def knn_dtw(dataset, neighbor=4):
    '''
    Implementation for K nearest neighbor with distance metric is dynamic time warping
    '''
    data = dataset.to_numpy()
    sh = data.shape
    data = data.reshape(sh[1], -1)

    #NOTE: The implementation of dtw take too long to calc, use fastdtw lib to calc in linear time
    # def dtw(a, b):
    #     an = a.size
    #     bn = b.size
    #     pointwise_distance = distance.cdist(a.reshape(-1, 1), b.reshape(-1, 1))
    #     cumdist = np.matrix(np.ones((an + 1, bn + 1)) * np.inf)
    #     cumdist[0, 0] = 0

    #     for ai in range(an):
    #         for bi in range(bn):
    #             minimum_cost = np.min([
    #                 cumdist[ai, bi + 1], cumdist[ai + 1, bi], cumdist[ai, bi]
    #             ])
    #             cumdist[ai + 1,
    #                     bi + 1] = pointwise_distance[ai, bi] + minimum_cost

    #     return cumdist[an, bn]

    from fastdtw import fastdtw as dtw

    n_neighbors = NearestNeighbors(n_neighbors=neighbor, algorithm='auto', leaf_size=30, metric=dtw)
    # fit
    n_neighbors.fit(data)

    dis, index = n_neighbors.kneighbors(data)
    print(dis)
    print(index)

    return dis, index


#NOTE:


def model_builder(input_dim=5, output_dim=1, target_timestep=1, dropout=0.1):
    from keras.layers import Input, Dense, TimeDistributed, LSTM, Bidirectional, Conv1D, Reshape, Concatenate
    from keras.models import Model

    input_tot = Input(shape=(None, input_dim))
    input_immediate_past = Input(shape=(output_dim, ))

    # conv = Conv1D(filters=16, kernel_size=2, strides=1, padding='same')
    # conv_out = conv(input_tot)
    # conv_2 = Conv1D(filters=32, kernel_size=3, padding='same')
    # conv_out_2 = conv_2(conv_out)
    # conv_3 = Conv1D(filters=64, kernel_size=4, padding='same')
    # conv_out_3 = conv_3(conv_out_2)

    rnn_1 = Bidirectional(
        LSTM(units=128, return_sequences=True, return_state=True, dropout=dropout, recurrent_dropout=dropout))
    rnn_out_1, forward_h, forward_c, backward_h, backward_c = rnn_1(input_tot)
    state_h = Concatenate(axis=-1)([forward_h, backward_h])
    state_c = Concatenate(axis=-1)([forward_c, backward_c])

    rnn_2 = LSTM(units=256, return_sequences=False, return_state=False, dropout=dropout, recurrent_dropout=dropout)
    rnn_out_2 = rnn_2(input_tot, initial_state=[state_h, state_c])

    #concat_out = Concatenate(axis=-1)([input_immediate_past, rnn_out_2])

    reshape_l = Reshape(target_shape=(target_timestep, -1))
    rnn_out = reshape_l(rnn_out_2)

    dense_3 = TimeDistributed(Dense(units=output_dim))
    output = dense_3(rnn_out)

    model = Model(inputs=[input_tot, input_immediate_past], outputs=output)

    #optimizer = SGD(lr=1e-6, momentum=0.9,decay=self.lr_decay,nesterov=True)
    #optimizer = RMSprop(learning_rate=5e-3)
    #optimizer = Adadelta(rho=0.95)
    #optimizer = Adam(learning_rate=5e-2,amsgrad=False)
    model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mape'])
    model.summary()
    return model


def train_model(model, x_train, x_train_past, y_train, batch_size, epochs, fraction=0.1, patience=0, save_dir=''):
    callbacks = []

    from keras.callbacks import EarlyStopping, ModelCheckpoint
    checkpoint = ModelCheckpoint(save_dir + f'best_model.hdf5', monitor='val_loss', verbose=1, save_best_only=True)
    callbacks.append(checkpoint)

    #early_stop = epochs == 250
    early_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    callbacks.append(early_stop)

    history = model.fit(x=[x_train, x_train_past],
                        y=y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_split=fraction)

    return model, history


if __name__ == "__main__":
    #reprocess_discharge(dat)
    #nearest_loc()

    dat_q = pd.read_csv('./RawData/discharge_daily.csv', index_col=0, header=0)
    #knn_dtw(dat_q)
    dat = pd.read_csv('./RawData/Kontum-daily.csv', header=0, index_col=0)

    #k_most_related(dat_q)

    dat['kongplong_q'] = dat_q['Kon_Plong']
    dat['dakmot_q'] = dat_q['Dak_Mot']

    from ProcessingData.reprocess_daily import extract_data

    x, y, scaler = extract_data(dat, cols_x=[3, 4, 5, 6, 7, 9, 10], cols_y=[5], mode='min_max')
    #x_immediate_past = dat.iloc[4:-2, 5].to_numpy()
    x_immediate_past = x[:, 4, 5]
    print(x_immediate_past.shape)
    model = model_builder(input_dim=7, output_dim=1, dropout=0)

    split_index = 1000
    x_train, y_train = x[:-split_index], y[:-split_index]
    x_test, y_test = x[-split_index:], y[-split_index:]
    x_past_train, x_past_test = x_immediate_past[:-split_index], x_immediate_past[-split_index:]

    train_model(model,
                x_train,
                x_past_train,
                y_train,
                batch_size=128,
                epochs=450,
                patience=50,
                save_dir='./Log/Adaptive/')

    import matplotlib.pyplot as plt

    y_pre = model.predict([x_test, x_past_test])
    #retransform
    mask = np.zeros(shape=dat.shape)
    test_shape = y_test.shape[0]

    mask[-test_shape:, 5] = y_test[:, 0, 0]
    actual_data = scaler.inverse_transform(mask)[-test_shape:, 5]

    mask[-test_shape:, 5] = y_pre[:, 0, 0]
    actual_predict = scaler.inverse_transform(mask)[-test_shape:, 5]

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