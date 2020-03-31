import pandas as pd 
import numpy as np



def normalize_data(dataframe,mode):
    if mode == 'l2':
        from sklearn.preprocessing import normalize
        return normalize(dataframe,axis=0)
    if mode == 'min_max':
        from sklearn.preprocessing import MinMaxScaler
        minmax = MinMaxScaler(feature_range=(0,1),copy=True) #save for retransform later
        minmax.fit(dataframe)
        data_norm = minmax.transform(dataframe)
        return data_norm
        # TODO: change later to return scaler to retransform 

def extract_data(dataframe,window_size=5,cols=[],mode='l2'):
    dataframe = dataframe.drop('time',axis=1)
    dataframe = dataframe.to_numpy()

    dataframe = normalize_data(dataframe,mode)

    xs = []
    ys = []
    for i in range(dataframe.shape[0] - window_size - 1):
        xs.append(dataframe[i:i+window_size,cols])
        ys.append(dataframe[i+window_size,[7,5]].reshape(1,2))

    return np.array(xs),np.array(ys)

def ed_extract_data(dataframe,window_size=5,cols=[],mode='l2'):
    dataframe = dataframe.drop('time',axis=1)
    dataframe = dataframe.to_numpy()

    dataframe = normalize_data(dataframe,mode)

    en_x = []
    de_x = []
    de_y = []

    for i in range(dataframe.shape[0] - window_size - 1):
        en_x.append(dataframe[i:i + window_size,cols])

        #decoder input is q and h of 'window-size' days before
        de_x.append(dataframe[i:i + window_size,[7,5]])
        de_y.append(dataframe[i + window_size,[7,5]].reshape(1,2))

    return np.array(en_x),np.array(de_x),np.array(de_y)

if __name__ == '__main__':
    dataframe = pd.read_csv('./RawData/Kontum-daily.csv',header=0,index_col=0)
    #print(dataframe.head())
    cols = list(range(8))
    #print(cols)
    en_x, de_x, de_y = ed_extract_data(dataframe=dataframe,cols=cols)
    print(en_x.shape)
    print(en_x[:5])
    print(de_x.shape)
    print(de_x[:5])
    print(de_y.shape)
    print(de_y[:5])