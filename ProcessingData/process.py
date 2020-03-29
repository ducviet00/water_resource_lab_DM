import pandas as pd 
import numpy as np
from tensorflow import keras
from tensorflow import data
from tensorflow import cast,float32,reshape


dtframe = pd.read_csv('./ProcessedData/Ban Don.csv')

dtframe= dtframe.drop('Time',axis=1)
#print(dtframe.head())
cols = ['H','Q']

def split_window_data(dtframe,window_size=6,cols=[]):
    dataset = data.Dataset.from_tensor_slices(
            cast(dtframe[cols].values,float32)
    )
     # +1 for the output
    dataset = dataset.window(size=window_size +1,shift=1,drop_remainder=True)
    # add additional dimension
    dataset = dataset.flat_map(lambda window: window.batch(window_size+1))
    dataset = dataset.shuffle(1000).map(lambda window : (window[:-1],reshape(window[-1:],(2,))))
    xs = []
    ys = []
    iterator = dataset.__iter__()
    done_looping = False
    while not done_looping:
        try:
            x , y = iterator.next()
            xs.append(x.numpy())
            ys.append(y.numpy())
        except StopIteration:
            done_looping = True
    return np.array(xs),np.array(ys)


if __name__ == '__main__':
    xs,ys = split_window_data(dtframe =dtframe,cols=cols)
    print(xs.shape)
    print(ys[:3])
