import pandas as pd
import numpy as np


def normalize_data(dataframe, mode):
    if mode == 'abs':
        from sklearn.preprocessing import MaxAbsScaler
        max_abs = MaxAbsScaler(copy=True)  #save for retransform later
        max_abs.fit(dataframe)
        data_norm = max_abs.transform(dataframe)

        return data_norm, max_abs

    if mode == 'robust':
        from sklearn.preprocessing import RobustScaler
        robust = RobustScaler(copy=True)  #save for retransform later
        robust.fit(dataframe)
        data_norm = robust.transform(dataframe)

        return data_norm, robust

    if mode == 'min_max':
        from sklearn.preprocessing import MinMaxScaler
        minmax = MinMaxScaler(feature_range=(0, 1), copy=True)  #save for retransform later
        minmax.fit(dataframe)
        data_norm = minmax.transform(dataframe)

        return data_norm, minmax
    if mode == 'std':
        from sklearn.preprocessing import StandardScaler
        stdscaler = StandardScaler(copy=True, with_mean=True, with_std=True)
        stdscaler.fit(dataframe)
        data_norm = stdscaler.transform(dataframe)

        return data_norm, stdscaler


def extract_data(dataframe, window_size=5, target_timstep=1, cols_x=[], cols_y=[], mode='std'):
    dataframe, scaler = normalize_data(dataframe, mode)

    xs = []
    ys = []

    for i in range(dataframe.shape[0] - window_size - target_timstep):
        xs.append(dataframe[i:i + window_size, cols_x])
        ys.append(dataframe[i + window_size:i + window_size + target_timstep,
                            cols_y].reshape(target_timstep, len(cols_y)))

    return np.array(xs), np.array(ys), scaler


def ed_extract_data(dataframe, window_size=5, target_timstep=1, cols_x=[], cols_y=[], mode='std'):
    dataframe, scaler = normalize_data(dataframe, mode)

    en_x = []
    de_x = []
    de_y = []

    for i in range(dataframe.shape[0] - window_size - target_timstep):
        en_x.append(dataframe[i:i + window_size, cols_x])

        #decoder input is q and h of 'window-size' days before
        de_x.append(dataframe[i + window_size - 1:i + window_size + target_timstep - 1,
                              cols_y].reshape(target_timstep, len(cols_y)))
        de_y.append(dataframe[i + window_size:i + window_size + target_timstep,
                              cols_y].reshape(target_timstep, len(cols_y)))

    en_x = np.array(en_x)
    de_x = np.array(de_x)
    de_y = np.array(de_y)
    de_x[:, 0, :] = 0

    return en_x, de_x, de_y, scaler


def atted_extract_data(dataframe, window_size=5, cols=[], mode='l2'):  #NOTE: unuse!!!
    dataframe, scaler = normalize_data(dataframe, mode)

    en_x = []
    de_x = []
    de_y = []

    for i in range(dataframe.shape[0] - 2 * window_size - 1):
        en_x.append(dataframe[i:i + window_size, cols])

        #decoder input is q and h of 'window-size' days before
        de_x.append(dataframe[(i + window_size - 1):(i + 2 * window_size - 1), [7, 5]])
        de_y.append(dataframe[(i + window_size):(i + 2 * window_size), [7, 5]])

    en_x = np.array(en_x)
    de_x = np.array(de_x)
    de_y = np.array(de_y)

    de_x[:, 0, :] = 0

    return en_x, de_x, de_y, scaler


def roll_data(dataframe, cols_x, cols_y, mode='min_max'):
    dataframe, scaler = normalize_data(dataframe, mode)
    #dataframe = dataframe.drop('time', axis=1)

    X = dataframe[:, cols_x]
    y = dataframe[:, cols_y]

    return X, y, scaler


def process_evaporation(full_dat, vapor_dat):
    full_dat['vapor'] = 0
    length = len(full_dat)

    full_dat.index = pd.to_datetime(full_dat['time'])
    full_dat = full_dat.drop('time', axis=1)

    vapor_dat.index = pd.to_datetime(vapor_dat['Time'])
    vapor_dat = vapor_dat.drop('Time', axis=1)

    vapor_dat['KonTum'] = vapor_dat['KonTum'] / 12

    print(full_dat.head())
    print(vapor_dat.head())
    print(length)
    #vapor_dat.to_csv('./RawData/KonTum.csv')
    import datetime
    for i in range(length):
        month = full_dat.index[i].month
        year = full_dat.index[i].year
        full_dat.iloc[i, 8] = vapor_dat.loc[datetime.datetime(year, month, 1), 'KonTum']
        if i < 10:
            print(vapor_dat.loc[datetime.datetime(year, month, 1), 'KonTum'])

    print(full_dat.head())
    full_dat.to_csv('./Kontum-daily.csv')


def plot_compare_model():
    dat = pd.read_csv('./Log/DataAnalysis/predict_val.csv', header=0)
    # dat_arima = pd.read_csv('./Log/Arima/arima.csv', header=0, index_col=0)

    # dat['arima_q'] = dat_arima['arima_q_hanoi']
    # dat['arima_h'] = dat_arima['arima_h_hanoi']

    # dat.to_csv('./Log/DataAnalysis/predict_val.csv', index=None)
    import matplotlib.pyplot as plt

    plt.plot(dat['real_q'], label='ground_truth')
    plt.plot(dat['ensemble_q'], label='ensemble')
    plt.plot(dat['rnn_cnn_q'], label='rnn_cnn')
    plt.plot(dat['en_de_q'], label='encoder_decoder')
    plt.legend(loc='best')
    plt.xlabel('Time')
    plt.ylabel('Q')
    plt.title('Các mô hình đề xuất')
    plt.savefig('./Log/DataAnalysis/our_model_q.png')
    plt.clf()

    plt.plot(dat['real_h'], label='ground_truth')
    plt.plot(dat['ensemble_h'], label='ensemble')
    plt.plot(dat['rnn_cnn_h'], label='rnn_cnn')
    plt.plot(dat['en_de_h'], label='encoder_decoder')
    plt.legend(loc='best')
    plt.xlabel('Time')
    plt.ylabel('H')
    plt.title('Các mô hình đề xuất')

    plt.savefig('./Log/DataAnalysis/our_model_h.png')
    plt.clf()

    plt.plot(dat['lstm_q'], label='lstm')
    plt.plot(dat['ann_q'], label='ann')
    plt.plot(dat['arima_q'], label='arima')
    plt.plot(dat['real_q'], label='ground_truth')

    plt.legend(loc='best')
    plt.xlabel('Time')
    plt.ylabel('Q')
    plt.title('Các mô hình hiện tại')
    plt.savefig('./Log/DataAnalysis/paper_model_q.png')
    plt.clf()

    plt.plot(dat['lstm_h'], label='lstm')
    plt.plot(dat['ann_h'], label='ann')
    plt.plot(dat['arima_h'], label='arima')
    plt.plot(dat['real_h'], label='ground_truth')

    plt.legend(loc='best')
    plt.xlabel('Time')
    plt.ylabel('H')
    plt.title('Các mô hình hiện tại')
    plt.savefig('./Log/DataAnalysis/paper_model_h.png')
    plt.clf()

    plt.plot(dat['real_q'], label='ground_truth')
    plt.plot(dat['ensemble_q'], label='ensemble')
    plt.plot(dat['rnn_cnn_q'], label='rnn_cnn')
    plt.plot(dat['en_de_q'], label='encoder_decoder')
    plt.plot(dat['lstm_q'], label='lstm')
    plt.plot(dat['ann_q'], label='ann')
    plt.plot(dat['arima_q'], label='arima')
    plt.legend(loc='best')
    plt.xlabel('Time')
    plt.ylabel('Q')
    plt.title('Mọi mô hình')
    plt.savefig('./Log/DataAnalysis/compare_q.png')
    plt.clf()

    plt.plot(dat['real_h'], label='ground_truth')
    plt.plot(dat['ensemble_h'], label='ensemble')
    plt.plot(dat['rnn_cnn_h'], label='rnn_cnn')
    plt.plot(dat['en_de_h'], label='encoder_decoder')
    plt.plot(dat['lstm_h'], label='lstm')
    plt.plot(dat['ann_h'], label='ann')
    plt.plot(dat['arima_h'], label='arima')
    plt.legend(loc='best')
    plt.xlabel('Time')
    plt.ylabel('H')
    plt.title('Mọi mô hình')
    plt.savefig('./Log/DataAnalysis/compare_H.png')
    plt.clf()


def plot_PM():
    import matplotlib.pyplot as plt

    gt = pd.read_csv('./RawData/PM/groundtruth.csv', header=None)
    pre = pd.read_csv('./RawData/PM/preds.csv', header=None)

    plt.plot(gt[1], label='ground_truth')
    plt.plot(pre[1], label='predict')

    plt.legend(loc='best')
    plt.xlabel('Thời gian')
    plt.ylabel('Lượng mưa')
    plt.title('Kết quả mô hình')
    plt.savefig('./Log/DataAnalysis/compare_pm.png')


if __name__ == '__main__':
    # full_dat = pd.read_csv('./RawData/Kontum-daily.csv', header=0, index_col=0)
    # vapor_dat = pd.read_csv('./RawData/KonTum.csv', header=0, index_col=None)

    # process_evaporation(full_dat, vapor_dat)

    # data = pd.read_csv('./RawData/Hanoi/Merge_HN.csv', header=0, index_col=0)
    # data = data.set_index('date')
    # data = data.drop(['date.1', 'date.2'], axis=1)
    # print(data.head())
    # data.to_csv('./RawData/Hanoi/Merge_HN.csv')
    # data = data.to_numpy()
    # print(data.shape)
    plot_PM()