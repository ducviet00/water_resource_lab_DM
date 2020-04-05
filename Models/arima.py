import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima_model import ARIMA
#from ProcessingData.correlation import data_group

def train(col):
    #data = data_group()
    data = pd.read_csv('./RawData/Kontum-daily.csv',header=0,index_col=0)
    data.index = pd.to_datetime(data['time'])

    X = data[col].values

    train, test = X[:-800], X[-800:]
    predict = []
    his = [x for x in train] #copy train set

    for i in range(len(test)):
        model = ARIMA(his,order=(5,1,0)) #roll forecasting
        model_fit = model.fit(disp=0)

        out = model_fit.forecast()

        yhat = out[0][0]
        predict.append(yhat)
        his.append(test[i]) #after predict 1 day, append the real value of that day to the training set

        print(f'Expect val: {test[i]}, Predict val: {yhat}')

    mae = mean_absolute_error(test,predict)
    print(f'MAE: {mae}')

    plt.plot(test,label='truth')
    plt.plot(predict,label='predict')
    plt.legend(loc='best')
    plt.savefig('./Log/Arima/predict_h.png')
    plt.show()

if __name__=='__main__':
    # import sys
    # import os
    # sys.path.append('../')
    # print(sys.path)
    train('water_level')

    
        