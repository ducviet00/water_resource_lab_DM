import pandas as pd 
import numpy as np



def normalize_data(dataframe,mode):
    if mode == 'l2':
        from sklearn.preprocessing import normalize
        return normalize(dataframe,axis=0), None
    if mode == 'min_max':
        from sklearn.preprocessing import MinMaxScaler
        minmax = MinMaxScaler(feature_range=(0,1),copy=True) #save for retransform later
        minmax.fit(dataframe)
        data_norm = minmax.transform(dataframe)
        
        return data_norm, minmax

def extract_data(dataframe,window_size=5,cols=[],mode='l2'):
    dataframe, scaler = normalize_data(dataframe,mode)

    xs = []
    ys = []
    
    for i in range(dataframe.shape[0] - window_size - 1):
        xs.append(dataframe[i:i+window_size,cols])
        ys.append(dataframe[i+window_size,[7,5]].reshape(1,2))

    return np.array(xs),np.array(ys), scaler

def ed_extract_data(dataframe,window_size=5,target_timstep=1,cols_x=[],cols_y=[],mode='l2'):
    dataframe, scaler = normalize_data(dataframe,mode)

    en_x = []
    de_x = []
    de_y = []

    for i in range(dataframe.shape[0] - window_size - target_timstep):
        en_x.append(dataframe[i:i + window_size,cols_x])

        #decoder input is q and h of 'window-size' days before
        de_x.append(dataframe[i + window_size - 1: i + window_size + target_timstep -1,cols_y].reshape(target_timstep,len(cols_y)))
        de_y.append(dataframe[i + window_size : i + window_size + target_timstep,cols_y].reshape(target_timstep,len(cols_y)))

    en_x = np.array(en_x)
    de_x = np.array(de_x)
    de_y = np.array(de_y)
    de_x[0,:,:] = 0
    
    return en_x,de_x,de_y, scaler

def atted_extract_data(dataframe,window_size=5,cols=[],mode='l2'):
    dataframe, scaler = normalize_data(dataframe,mode)

    en_x = []
    de_x = []
    de_y = []

    for i in range(dataframe.shape[0] - 2* window_size - 1):
        en_x.append(dataframe[i:i + window_size,cols])

        #decoder input is q and h of 'window-size' days before
        de_x.append(dataframe[(i + window_size -1) : (i + 2*window_size-1) ,[7,5]])
        de_y.append(dataframe[(i + window_size) : (i + 2*window_size) ,[7,5]])

    en_x = np.array(en_x)
    de_x = np.array(de_x)
    de_y = np.array(de_y)
    
    de_x[0,:,:] = 0

    return en_x,de_x,de_y,scaler

if __name__ == '__main__':
    dataframe = pd.read_csv('./RawData/Kontum-daily.csv',header=0,index_col=0)
    dataframe = dataframe.drop('time',axis=1)
    dataframe = dataframe.to_numpy()
    # import matplotlib.pyplot as plt 
    # plt.plot(dataframe[:,7])
    # plt.show()
    # print(np.argmax(dataframe[:,5],axis=0))
    # print(dataframe.shape)
    # print(dataframe[0])
    #print(dataframe.head())
    cols = list(range(8))
    #print(cols)
    en_x, de_x, de_y = ed_extract_data(dataframe=dataframe,cols_x=cols,cols_y=[5],target_timstep=5,mode='min_max')
    
    print(de_x.shape)
    #de_y = scaler.inverse_transform(de_y.reshape(de_y.shape[0],2,))
    print(de_x[:5])

    print(en_x[:5])

    print(de_y[:5])