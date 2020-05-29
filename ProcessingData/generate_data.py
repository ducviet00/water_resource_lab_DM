import pandas as pd
import numpy as np


def mean_generate_data(data, mean_frame=2):
    nrow = len(data) - mean_frame + 1
    ncol = len(data.iloc[0])
    gen_dat = pd.DataFrame(np.zeros(shape=(nrow, ncol)))
    gen_dat.columns = data.columns

    for i in range(nrow):
        gen_dat.iloc[i] = data.iloc[i:i + mean_frame].mean(axis=0)

    gen_dat = gen_dat.round(4)
    print(gen_dat.head())
    gen_dat.to_csv('./ProcessedData/GeneratedData/mean_gen_1.csv', index=None)


if __name__ == "__main__":
    data = pd.read_csv('./RawData/Kontum-daily.csv', header=0, index_col=0)
    data = data.drop('time', axis=1)
    mean_generate_data(data, mean_frame=3)
