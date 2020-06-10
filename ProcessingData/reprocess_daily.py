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


def roll_data(link, window_size, target_col, cols_x, cols_y):
    dataframe = pd.read_csv(link, header=0, index_col=0)
    dataframe = dataframe.drop('time', axis=1)
    dataframe = dataframe.to_numpy()

    X = dataframe[:, cols_x]
    y = dataframe[:, cols_y]

    return X, y


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


if __name__ == '__main__':
    full_dat = pd.read_csv('./RawData/Kontum-daily.csv', header=0, index_col=0)
    vapor_dat = pd.read_csv('./RawData/KonTum.csv', header=0, index_col=None)

    process_evaporation(full_dat, vapor_dat)
