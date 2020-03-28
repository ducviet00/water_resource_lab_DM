import pandas as pd 
from tensorflow import keras
from tensorflow import data
from tensorflow import cast,float32


dtframe = pd.read_csv('./ProcessedData/Ban Don.csv')

dtframe= dtframe.drop('Time',axis=1)
print(dtframe.head())
cols = ['H','Q']

def split_window_data(dtframe,window_size=6,batch_size=32,cols=[]):
    dataset = data.Dataset.from_tensor_slices(
        (
            cast(dtframe[cols].values,float32),
        )
    )
     # +1 for the output
    dataset = dataset.window(size=window_size +1,shift=1,drop_remainder=True)
    # add additional dimension
    dataset = dataset.flat_map(lambda window: window.batch(window_size+1))
    dataset = dataset.shuffle(1000).map(lambda window : (window[:-1],window[-1:]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


if __name__ == '__main__':
    dtset = split_window_data(dtframe,cols)
    print(list(dtset.__iter__())[:1])
